#include "cl_bp_utils.h"
#include "cl_error_handlers.h"
#include <stdio.h>
#include <math.h>

// ascii has this bit set for lowercase letters
#define LOWER_MASK 0x20

cl_mem createImageBuffer(cl_context context, char is_host_readable, char is_array, const cl_image_format* img_format, const size_t img_size[3])
{
	cl_int clErr;
	cl_mem_flags flags = is_host_readable ? (CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY):(CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);

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

	printf("Creating %zu*%zu*%zu buffer.", img_size[0], img_size[1], img_size[2]);
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
char isArgMetadataValid(char* metadata)
{
	// valid amount of channels
	if(metadata[2] <= '0' || metadata[2] > '4')
		return 0;
	// valid r/w type and storage type combination
	switch (metadata[1])
	{
	case 'c':
	case 's':
	case 'i':
		if(metadata[0] == 'u' || metadata[0] == 'f')
			return 1;
	case 'C':
	case 'S':
	case 'I':
		if(metadata[0] == 'i')
			return 1;
	case 'f':
	case 'h':
	case 'F':
	case 'H':
		if(metadata[0] == 'f' && ((metadata[1] | LOWER_MASK) != 'i'))	// no int32 backed type for floating point
			return 1;
	default:
		return 0;
	}
}

cl_channel_type getTypeFromMetadata(const char* metadata)
{
	if(metadata[0] == 'f')
	{
		switch(metadata[1])
		{
		case 'F':
		case 'f':
			return CL_FLOAT;
		case 'H':
		case 'h':
			return CL_HALF_FLOAT;
		case 'c':
			return CL_UNORM_INT8;
		case 's':
			return CL_UNORM_INT16;
		case 'C':
			return CL_SNORM_INT8;
		case 'S':
			return CL_SNORM_INT16;
		}
		//warning fix/error trap
		return 0;
	}
	//else
	switch(metadata[1])
	{
	case 'c':
		return CL_UNSIGNED_INT8;
	case 's':
		return CL_UNSIGNED_INT16;
	case 'i':
		return CL_UNSIGNED_INT32;
	case 'C':
		return CL_SIGNED_INT8;
	case 'S':
		return CL_SIGNED_INT16;
	case 'I':
		return CL_SIGNED_INT32;
	}
	//warning fix/error trap
	return 0;
}

cl_channel_order getOrderFromMetadata(const char* metadata)
{
	//only powers of 2 channels can be used for write buffers so 3 gets promoted to 4
	switch(metadata[2])
	{
	case '1':
		return CL_R;
	case '2':
		return CL_RG;
	}
	return CL_RGBA;
}

// in can be NULL if mode is EXACT or SINGLE
void calcSizeByMode(const size_t* in, const RangeData* range, size_t* out)
{

	if(!in && range->mode != EXACT && range->mode != SINGLE)
	{
		out[0] = 0;	// if you got here you probably didn't specify something correctly
		return;
	}

	switch (range->mode)
	{
	case EXACT:
		out[0] = range->param[0];
		out[1] = range->param[1];
		out[2] = range->param[2];
		return;
	case SINGLE:	//TODO: make this check the target device to find out how many threads to a hardware compute unit
		out[0] = 1;
		out[1] = 1;
		out[2] = 1;
		return;
	case REL:
		out[0] = in[0] + range->param[0];
		out[1] = in[1] + range->param[1];
		out[2] = in[2] + range->param[2];
		return;
	case DIAG:
		out[0] = range->param[0];
		out[1] = ((int)sqrt(in[0]*in[0] + in[1]*in[1]) + range->param[1]) & -2;	//diagonal length truncated down to even
		out[2] = in[2] + range->param[2];
		return;
	case DIVIDE:
		out[0] = in[0] / range->param[0];
		out[1] = in[1] / range->param[1];
		out[2] = in[2] / range->param[2];
		return;
	}
	out[0]=0;	// if you got here you probably forgot to implement a mode
}

//TODO: add ifdefs for OpenCL versions to the following 3 helper functions
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

// currently assumes number of channels is the only thing important, NOT posistioning or ordering
unsigned char getChannelCount(cl_channel_order order)
{	// not certain I interpreted the _x channel counts right but it'll do for now until I have a problem
	switch(order)
	{
	case CL_R:
	case CL_A:
	case CL_Rx:
	case CL_INTENSITY:
	case CL_LUMINANCE:
//	case CL_DEPTH:
		return 1;
	case CL_RG:
	case CL_RA:
	case CL_RGx:
		return 2;
	case CL_RGB:
	case CL_RGBx:
//	case CL_sRGB:
//	case CL_sRGBx:
		return 3;
	case CL_RGBA:
	case CL_BGRA:
	case CL_ARGB:
//	case CL_ABGR:
//	case CL_sRGBA:
//	case CL_sBGRA:
		return 4;
	default:
		return 0;
	}
}

unsigned char getChannelWidth(char metadata_type)
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
		fprintf(stderr, "\nWarning: possible arg/read type mismatch\n found:%c, expected:%c\n", found_rw_type, metadata[0]);

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
				fputs("\nWarning: possible signedness mismatch, may lead to errors based on input range assumptions\n", stderr);
		}
		break;
	case 'u':
	case 'i':
		if(isHalfOrFloat)
			fputs("\nWarning: possible signed/unsigned integer read attempt from float image\n", stderr);
		else if(isNotSameSignedness)
			fputs("\nWarning: possible integer signedness mismatch\n", stderr);
		break;
	default:
		break;
	}

	//check minimum expected channels
	unsigned char expected_channels = metadata[2] - '0';
	if(expected_channels > 4)
		fputs("\nWarning: non-conforming metadata found\n", stderr);
	else
	{
		char found_channels = getChannelCount(ref_format.image_channel_order);
		if(found_channels < expected_channels)
			fputs("\nWarning: less channels available to read than expected\n", stderr);
	}
}

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
