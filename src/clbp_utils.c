#include "clbp_utils.h"
#include "clbp_error_handling.h"
#include "cl_error_handlers.h"
#include <stdio.h>
#include <math.h>

// ascii has this bit set for lowercase letters
//#define LOWER_MASK 0x20

cl_mem createImageBuffer(cl_context context, char force_host_readable, char is_array, const cl_image_format* img_format, const size_t img_size[3])
{
	cl_int clErr;
	cl_mem_flags flags = force_host_readable ? (CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY):(CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);

	cl_image_desc image_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE1D,
		.image_width = img_size[0],
		.image_height = img_size[1],
		.image_depth = img_size[2],
		.image_array_size = 1,
		.image_row_pitch = 0,
		.image_slice_pitch = 0,
		.num_mip_levels = 0,
		.num_samples = 0,
		.buffer= NULL
	};

	// auto mem type discovery from array flag and provided dimensions
	// also fixes image descriptor where appropriate
	if(img_size[2] > 1)
	{
		if(is_array)
		{
			image_desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
			image_desc.image_depth = 1;
			image_desc.image_array_size = img_size[2];
		}
		else
			image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
	}
	else if(img_size[1] > 1)
	{
		if(is_array)
		{
			image_desc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
			image_desc.image_height = 1;
			image_desc.image_array_size = img_size[1];
		}
		else
			image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;

	}

	printf("Creating %zu*%zu*%zu buffer with format %i.", img_size[0], img_size[1], img_size[2], getChannelCount(img_format->image_channel_order));
	cl_mem img_buffer = clCreateImage(context, flags, img_format, &image_desc, NULL, &clErr);
	handleClError(clErr, "clCreateImage");
/*
	*out_data = (char*)malloc(img_size[0] * img_size[1]*4);
	if(!out_data)
	{
		perror("failed to allocate output buffer");
		exit(1);
	}
*/
	return img_buffer;
}

// validates metadata[0 thru 2] formating and returns true if valid
char isArgMetadataValid(char const metadata[static 3])
{
	// valid amount of channels
	if(metadata[2] <= '0' || metadata[2] > '4')
		return 0;
	// valid r/w type and storage type combination
	switch (metadata[0])
	{
	case 'u':	// unsigned int r/w
	case 'i':	// signed int r/w
		// check expected width validity
		switch(metadata[1])
		{
		case 'c':	// char		 8-bit
		case 's':	// short	16-bit
		case 'i':	// int		32-bit
			return 1;
		default:
			return 0;
		}
	case 'f':	// float r/w
	case 'h':	// half r/w
		// check expected range validity
		switch(metadata[1])
		{
		case 'u':	// unsigned normalized	[ 0.0, 1.0]
		case 's':	// signed normalized	[-1.0, 1.0]
		case 'f':	// full range			[-Inf, Inf]
			return 1;
		}
	}
	return 0;
}

// checks provided channel type against one requested via the metadata, while some
// mismatches could result in undefined behavior, others are just likely not what
// you intended if you specified the metadata correctly, such as requesting a read
// from a full range float but providing a signed normalized value instead. This is
// primarily just to warn you that you might have made a mistake, but if it was
// intentional, you can safely ignore the warning that will follow
char isMatchingChannelType(const char* metadata, cl_channel_type type)
{
	switch(type)
	{
	case CL_UNSIGNED_INT8:
		return metadata[0] == 'u' && metadata[1] == 'c';
	case CL_UNSIGNED_INT16:
		return metadata[0] == 'u' && metadata[1] == 's';
	case CL_UNSIGNED_INT32:
		return metadata[0] == 'u' && metadata[1] == 'i';
	case CL_SIGNED_INT8:
		return metadata[0] == 'i' && metadata[1] == 'c';
	case CL_SIGNED_INT16:
		return metadata[0] == 'i' && metadata[1] == 's';
	case CL_SIGNED_INT32:
		return metadata[0] == 'i' && metadata[1] == 'i';
	case CL_SNORM_INT8:
	case CL_SNORM_INT16:
		if(metadata[1] != 's')
			return 0;
	case CL_HALF_FLOAT:
		if(metadata[0] == 'h')
			return 1;
	case CL_FLOAT:
		return metadata[0] == 'f';
	}
	// all the weird/extension only ones are unsigned normalized
	return metadata[1] == 'u' && (metadata[0] == 'h' || metadata[0] == 'f');
}

/*/ if the channel type has a restricted order, returns the order that best fits normal types
inline enum clChannelOrder isChannelTypeRestrictedOrder(enum clChannelType const type)
{
	switch(type)
	{
	case CLBP_UNORM_SHORT_565:
	case CLBP_UNORM_SHORT_555:
	case CLBP_UNORM_INT_101010:
		return CLBP_RGBx;
	case CLBP_UNORM_INT_101010_2:
		return CLBP_RGBA;
	//case CLBP_UNORM_INT_2_101010_EXT:
	//	return CLBP_ARGB;
	}
	return 0;
}*/

// returns the difference in number of channels provided vs requested,
char ChannelOrderDiff(char ch_cnt_data, cl_channel_order order)
{
	return getChannelCount(order) - (ch_cnt_data - '0');
}

// This is a massive oversimplification since NDRanges aren't capped at 3, but
// that's all I expect to ever need from this and it makes implementation much easier
// plus it's the minimum required upper limit for non-custom device types in the spec
// so it's the maximum reliably portable value
void calcSizeByMode(Size3D const* ref_arr, RangeData const* range_arr, Size3D* dest_arr, int count, clbp_Error* e)
{
	Size3D const* ref_size;
	RangeData const* curr_range;
	int32_t in[3], out[3];

	for(int i = 0; i < count; ++i)
	{
		curr_range = &range_arr[i];

		// safe to leave out if CLBP_RM_EXACT mode ranges are always set to a valid dummy ref_idx, ie 0.
		// mostly matters for hardcoded args, shouldn't matter for stages in manifest since unless they're the very first,
		// they can't reference out of range
	//	if(curr_range->mode != CLBP_RM_EXACT)
		{
			ref_size = &ref_arr[curr_range->ref_idx];
			// C promotion prevention
			in[0] = ref_size->d[0];
			in[1] = ref_size->d[1];
			in[2] = ref_size->d[2];
		}

		int16_t const* param = curr_range->param;
		// modes that don't use the reference don't need to check it.
		switch(curr_range->mode)
		{
		case CLBP_RM_EXACT:
			out[0] = param[0];
			out[1] = param[1];
			out[2] = param[2];
			break;
	/*	case SINGLE:	//TODO: make this check the target device to find out how many threads to a hardware compute unit
			out[0] = 1;
			out[1] = 1;
			out[2] = 1;
			break;
	*/	case CLBP_RM_ADD_SUB:
			out[0] = in[0] + param[0];
			out[1] = in[1] + param[1];
			out[2] = in[2] + param[2];
			break;
		case CLBP_RM_DIAGONAL:
			out[0] = param[0];
			out[1] = ((int)sqrt(in[0]*in[0] + in[1]*in[1]) + param[1]) & -2;	//diagonal length truncated down to even
			out[2] = in[2] + param[2];
			break;
		case CLBP_RM_DIVIDE:
			out[0] = in[0] / param[0];
			out[1] = in[1] / param[1];
			out[2] = in[2] / param[2];
			break;
		case CLBP_RM_MULTIPLY:
			out[0] = in[0] * param[0];
			out[1] = in[1] * param[1];
			out[2] = in[2] * param[2];
			break;
		case CLBP_RM_ROW:
			out[0] = param[0];
			out[1] = in[1] + param[1];
			out[2] = param[2];
			break;
		case CLBP_RM_COLUMN:
			out[0] = in[0] + param[0];
			out[1] = param[1];
			out[2] = param[2];
			break;
		default:	// if you got here you probably forgot to finish implementing a mode
			*e = (clbp_Error){.err_code = CLBP_INVALID_RANGEMODE, .detail = NULL + i};
			return;
		}
		// if any of the dimensions went non-positive, something about the configuration is wrong and must be aborted
		for(int j = 0; j < 3; ++j)
		{
			if(out[j] <= 0)
			{
				*e = (clbp_Error){.err_code = CLBP_INVALID_SIZE3D, .detail = NULL + i};
				return;
			}
			dest_arr[i].d[j] = out[j];
		}
	}
}
/*
char getDeviceRWType(cl_channel_type type)
{
	switch(type)
	{
	case CL_SNORM_INT8:
	case CL_SNORM_INT16:
	case CL_UNORM_INT8:
	case CL_UNORM_INT16:
	case CL_UNORM_SHORT_565:
	case CL_UNORM_SHORT_555:
	case CL_UNORM_INT_101010:
	case CL_HALF_FLOAT:
	case CL_FLOAT:
//	case CL_UNORM_INT_101010_2:
		return 'f';
	case CL_SIGNED_INT8:
	case CL_SIGNED_INT16:
	case CL_SIGNED_INT32:
		return 'i';
	case CL_UNSIGNED_INT8:
	case CL_UNSIGNED_INT16:
	case CL_UNSIGNED_INT32:
		return 'u';
	default:
		return '?';
	}
}

char getArgStorageType(cl_channel_type type)
{	//TODO: no clue if this is right with how the packed types work since official documentation doesn't specify
	switch(type)
	{
	case CL_UNORM_SHORT_565:
	case CL_UNORM_SHORT_555:
	case CL_UNORM_INT8:
	case CL_UNSIGNED_INT8:
		return 'c';
	case CL_SNORM_INT8:
	case CL_SIGNED_INT8:
		return 'C';
	case CL_UNORM_INT_101010:
//	case CL_UNORM_INT_101010_2:
	case CL_UNORM_INT16:
	case CL_UNSIGNED_INT16:
		return 's';
	case CL_SNORM_INT16:
	case CL_SIGNED_INT16:
		return 'S';
	case CL_SIGNED_INT32:
		return 'I';
	case CL_UNSIGNED_INT32:
		return 'i';
	case CL_HALF_FLOAT:
		return 'H';
	case CL_FLOAT:
		return 'F';
	default:
		return '?';
	}
}
*/
//TODO: currently assumes number of channels is the only thing important, NOT posistioning or ordering
// this is likely wrong for the case of RA, INTENSITY, LUMINANCE, and DEPTH, however none of those should
// be generated as output buffers which is the only time this should matter so it's not high priority to fix
uint8_t getChannelCount(cl_channel_order order)
{
	uint8_t const ch_cnts_LUT[] = {
		1,	//CL_R
		1,	//CL_A
		2,	//CL_RG
		2,	//CL_RA
		3,	//CL_RGB
		4,	//CL_RGBA
		4,	//CL_BGRA
		4,	//CL_ARGB
		1,	//CL_INTENSITY
		1,	//CL_LUMINANCE
		2,	//CL_Rx
		3,	//CL_RGx
		4,	//CL_RGBx
		1,	//CL_DEPTH
		3,	//CL_sRGB
		4,	//CL_sRGBx
		4,	//CL_sRGBA
		4,	//CL_sBGRA
		4,	//CL_ABGR
	};
	// range checked lookup, cl_channel_order is an unsigned type
	order -= CLBP_OFFSET_CHANNEL_ORDER;
	return (order < sizeof(ch_cnts_LUT)) ? ch_cnts_LUT[order] : 0;
}

//returns the size in bits of the largest channel for the type
uint8_t get4ChannelWidths(cl_channel_type type)
{
	uint8_t const widths_LUT[] = {
		4,	//SNORM_INT8
		8,	//SNORM_INT16
		4,	//UNORM_INT8
		8,	//UNORM_INT16
		2,	//UNORM_SHORT_565
		2,	//UNORM_SHORT_555
		4,	//UNORM_INT_101010
		4,	//SIGNED_INT8
		8,	//SIGNED_INT16
		16,	//SIGNED_INT32
		4,	//UNSIGNED_INT8
		8,	//UNSIGNED_INT16
		16,	//UNSIGNED_INT32
		8,	//HALF_FLOAT
		16,	//FLOAT
		0,	//RESERVED
		4,	//UNORM_INT_101010_2
	};

	type -= CLBP_OFFSET_CHANNEL_TYPE;
	return (type < sizeof(widths_LUT)) ? widths_LUT[type] : 0;
}

cl_channel_order getOrderFromChannelCnt(uint8_t count)
{
	uint8_t const ch_order_off[] = {0, 2, 4, 5};	//{CL_R, CL_RG, CL_RGB, CL_RGBA}
	--count;
	return count < 4 ? (cl_channel_order)CLBP_OFFSET_CHANNEL_ORDER + ch_order_off[count] : 0;
}

// get the minimum bytes per pixel allocation size for reading a given format of output buffer to the host
uint8_t getPixelSize(cl_image_format format)
{
	switch(format.image_channel_data_type)
	{
	case CLBP_UNORM_SHORT_565:
		return 3;	// the in-place conversion to 8-bit values will expand it from 2 bytes to 3
	case CLBP_UNORM_SHORT_555:
		return 3 + (format.image_channel_order != CL_RGB);	// 3 channel expands to 3 bytes, 4 channel expands to 4
	case CLBP_UNORM_INT_101010:
	case CLBP_UNORM_INT_101010_2:
		return 4;
	}
	
	return (getChannelCount(format.image_channel_order) * get4ChannelWidths(format.image_channel_data_type)) >> 2;
}

char isChannelTypePacked(cl_channel_type const type)
{	// bit vector where each bit index corresponds to that channel type being a packed type
	return (0b10000000001110000 >> (type - CLBP_OFFSET_CHANNEL_TYPE)) & 1;
}

char isChannelTypeSigned(cl_channel_type const type)
{	// bit vector where each bit index corresponds to that channel type being a signed type
	return (0b00110001110000011 >> (type - CLBP_OFFSET_CHANNEL_TYPE)) & 1;
}
/*
uint8_t getChannelWidth(char metadata_type)
{
	switch(metadata_type | LOWER_MASK)
	{
	case 'c':
		return 1;
	case 's':
	case 'h':
		return 2;
	case 'i':
	case 'f':
		return 4;
	default:
		return 0;
	}
}

// this only issues warnings to the user since they could easily have misnamed it
// and it isn't required data unlike on writes that need new textures
void verifyReadArgTypeMatch(cl_image_format ref_format, char* metadata)
{
	char found_rw_type = getDeviceRWType(ref_format.image_channel_data_type);

	if(found_rw_type != metadata[0])
		fprintf(stderr, "Warning: possible arg/read type mismatch\n\tfound:%c, expected:%c\n", found_rw_type, metadata[0]);

	char found_storage_type = getArgStorageType(ref_format.image_channel_data_type);
	char isHalfOrFloat = ((metadata[1] | LOWER_MASK) == 'f') || ((metadata[1] | LOWER_MASK) == 'h');
	char isNotSameSignedness = (metadata[1] ^ found_storage_type) & LOWER_MASK;
	switch(metadata[0])
	{
	case 'f':
		// 'f' is arbitrary floating point and can probably accept anything
		if(!isHalfOrFloat)
		{
			// case/signedness mis-match, can lead to errors based on input range assumptions
			// signed types translate to -1.0 to 1.0, and unsigned translate to 0.0 to 1.0
			if(isNotSameSignedness)
				fputs("Warning: possible signedness mismatch, may lead to errors based on input range assumptions\n", stderr);
		}
		break;
	case 'u':
	case 'i':
		if(isHalfOrFloat)
			fputs("Warning: possible signed/unsigned integer read attempt from float image\n", stderr);
		else if(isNotSameSignedness)
			fputs("Warning: possible integer signedness mismatch\n", stderr);
	}

	//check minimum expected channels
	uint8_t expected_channels = metadata[2] - '0';
	if(expected_channels > 4)
		fputs("Warning: non-conforming metadata found\n", stderr);
	else
	{
		char found_channels = getChannelCount(ref_format.image_channel_order);
		if(found_channels < expected_channels)
			fputs("Warning: less channels available to read than expected\n", stderr);
	}
}*/
/*
// converts an end relative arg index to a pointer to the referenced TrackedArg with error checking
TrackedArg* getRefArg(const ArgTracker* at, uint16_t rel_ref)
{
	// this value should always be <= 0 since it's indexing backwards
	// due to being defined as a uint16 cast to an int32 and then negated, this should always be the case
	int32_t ref_arg_idx = at->args_cnt -(int32_t)rel_ref;

	if(ref_arg_idx < 0)
	{
		fputs("\nIndexed past start of argTracker array", stderr);
		exit(1);
	}

	return &(at->args[ref_arg_idx]);
}
*/