#include "cl_boilerplate.h"
#include "cl_error_handlers.h"
#include "stb_image.h"

// ascii has this bit set for lowercase letters
#define LOWER_MASK 0x20
// attempts to get the first available GPU or if none available CPU
//TODO: actually implement multiple attempts to find a GPU, currently just takes the first device of the first platform
cl_device_id getPreferredDevice()
{
	cl_platform_id platform;
	cl_device_id device;
	cl_int clErr;

	// use the first platform
	clErr = clGetPlatformIDs(1, &platform, NULL);
	handleClError(clErr, "clGetPlatformIDs");

	// use the first device
	clErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	handleClError(clErr, "clGetDeviceIDs");

	return device;
}

//names is a NULL pointer terminated char* array of the kernel function names/filenames without file extensions
cl_uint buildKernelsFromSource(cl_context context, cl_device_id device, const char* src_dir, const char** names, const char* args, cl_kernel* kernels, cl_uint max_kernels)
{
	cl_int clErr;
	char str_buff[1024];

	// count number of names in NULL terminated char* array
	int k_src_cnt = 0;
	while(names[k_src_cnt] != NULL)
		++k_src_cnt;

	// allocate pointer array for kernel sources
	char** k_srcs = (char**)malloc(sizeof(char*) * k_src_cnt);
	if(k_srcs == NULL)
	{
		perror("\nCouldn't allocate kernel source array");
		exit(1);
	}

	// Read kernel source file and place content into buffer
	for(int i=0; i<k_src_cnt; ++i)
	{
		snprintf(str_buff, sizeof(str_buff)-1, "%s%s.cl", src_dir, names[i]);

		FILE* k_src_handle = fopen(str_buff, "r");
		if(k_src_handle == NULL)
		{
			fputs(str_buff, stderr);
			perror("\nCouldn't find the kernel program file");
			exit(1);
		}
		// get rough program size and allocate string
		fseek(k_src_handle, 0, SEEK_END);
		long k_src_size = ftell(k_src_handle);
		rewind(k_src_handle);

		k_srcs[i] = (char*)malloc(k_src_size + 1);
		if(k_srcs[i] == NULL)
		{
			perror("\nCouldn't allocate kernel source string");
			exit(1);
		}
		// program may become smaller due to line endings being partially stripped on read
		k_src_size = fread(k_srcs[i], sizeof(char), k_src_size, k_src_handle);
		fclose(k_src_handle);
		// terminate the string so we don't have to track sizes
		k_srcs[i][k_src_size] = '\0';
	}

	// Create program from file
	cl_program program = clCreateProgramWithSource(context, k_src_cnt, (const char**)k_srcs, NULL, &clErr);
	handleClError(clErr, "clCreateProgramWithSource");

	// done with the sources
	for(int i=0; i<k_src_cnt; ++i)
		free(k_srcs[i]);
	free(k_srcs);

	// Build program
	clErr = clBuildProgram(program, 1, &device, args, NULL, NULL);
	handleClBuildProgram(clErr, program, device);

	// warn user if they specified a program with more kernels than they provided space for
	size_t kernel_cnt;
	clErr = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS, sizeof(size_t), &kernel_cnt, NULL);
	handleClError(clErr, "clGetProgramInfo");
	if(kernel_cnt > max_kernels)
		fputs("\nWARNING: more kernels exist in program than specified max_kernels", stderr);
	
	// create kernels from the built program
	cl_uint kernels_ret;
	clErr = clCreateKernelsInProgram(program, max_kernels, kernels, &kernels_ret);
	handleClError(clErr, "clCreateKernelsInProgram");

	// done with program, safe to release it now so we don't need to track it
	// It won't be fully released yet since the kernels have references to it
	clReleaseProgram(program);
	handleClError(clErr, "clReleaseProgram");

	return kernels_ret;
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
	case 'F':
	case 'H':
		if(metadata[0] == 'f' && ((metadata[1] | LOWER_MASK) != 'i'))	// no int32 backed type for floating point
			return 1;
	default:
		return 0;
	}
}

cl_channel_type getTypeFromMetadata(const char* metadata, char isHostReadable)
{
	if(metadata[0] != 'f')
	{
		//TODO: if I ever migrate from stbi to libPNG etc, this will have to be fixed for better bit depth
		if(isHostReadable)
		{
			if(metadata[1] & LOWER_MASK)
				return CL_UNSIGNED_INT8;
			return CL_SIGNED_INT8;
		}
		
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
	//else
	if(isHostReadable)
	{
		if(metadata[1] & LOWER_MASK)
			return CL_UNORM_INT8;
		return CL_SNORM_INT8;
	}

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

	//size_t* ref_size;
	if(!in && range->mode != EXACT && range->mode != SINGLE)
	{
		out[0] = 0;	// if you got here you probably didn't specify something correctly
		return;
	}

	// default to relative {0,0,0} if range param is NULL
	//TODO: might change this behavior if it turns out having the values default to specific values is convenient
	/*if(!range->param && range->mode != SINGLE)
	{
		out[0] = in[0];
		out[1] = in[1];
		out[2] = in[2];
		return;
	}*/

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

// returns the max number of bytes needed for reading out of any of the host readable buffers
//TODO: add support for returning a list of host readable buffers
void setKernelArgs(cl_context context, const QStaging* stage, cl_kernel kernel, ArgTracker* at)
{
	cl_int clErr;
	// get the count of how many args the kernel has to iterate over
	cl_uint arg_cnt;
	clErr = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &arg_cnt, NULL);
	handleClError(clErr, "clGetKernelInfo");

	for(cl_uint j=0; j < arg_cnt; ++j)
	{
		printf("[%i] ", j);
		cl_kernel_arg_access_qualifier arg_access;
		clErr = clGetKernelArgInfo(kernel, j, CL_KERNEL_ARG_ACCESS_QUALIFIER, sizeof(cl_kernel_arg_access_qualifier), &arg_access, NULL);
		handleClError(clErr, "clGetKernelArgInfo");

		// I attach type data in a similar style to Hungarian notation to the names so that the expected type backing of
		// the image is stored with the kernel itself and can be interpreted here by querying the name.
		// [0] == [f/u/i] the read/write type used with it
		// [1] == [F/H/C/S/I/f/h/c/s/i]	the type and signedness of the expected texture, uppercase is signed, lowercase is unsigned
		// [2] == [1/2/3/4] how many channels it expects
		// [4] == [_/n] 'n' for read/write buffers that are new buffers, otherwise not relevant
		char arg_metadata[64];	// although only 4 entries are needed, reading the name will fail if there's not enough room for the whole name
		clErr = clGetKernelArgInfo(kernel, j, CL_KERNEL_ARG_NAME, 64, arg_metadata, NULL);
		handleClError(clErr, "clGetKernelArgInfo");

		// current arg staging data being processed
		ArgStaging* this_s_arg = &(stage->args[j]);

		TrackedArg* ref_arg = getRefArg(at, this_s_arg->rel_ref);

		char isValid = isArgMetadataValid(arg_metadata);

		// write-only and certain r/w arguments create new mem objects to write their outputs to
		//NOTE: theoretically you might have an uneeded output that you leave unassigned but I'm not supporting that, just rewrite it without it instead
		if(arg_access == CL_KERNEL_ARG_ACCESS_WRITE_ONLY || (arg_access == CL_KERNEL_ARG_ACCESS_READ_WRITE && arg_metadata[3] == 'n'))
		{
			if(!isValid)
			{
				fputs("\nCouldn't parse argument metadata, can't create image object", stderr);
				exit(1);
			}

			if(at->args_cnt >= at->max_args)
			{
				fputs("\nNot enough room in ArgTracker.args array", stderr);
				exit(1);
			}

			TrackedArg* this_t_arg = &(at->args[at->args_cnt]);

			this_t_arg->format.image_channel_data_type = getTypeFromMetadata(arg_metadata, this_s_arg->is_host_readable);
			this_t_arg->format.image_channel_order = getOrderFromMetadata(arg_metadata);
			calcSizeByMode(ref_arg->size, &this_s_arg->range, this_t_arg->size);

			// if this is a host readable output, we need to see if the size in bytes is bigger than any previous args so the
			// final read buffer can be allocated large enough
			if(this_s_arg->is_host_readable)
			{
				size_t size_in_bytes = this_t_arg->size[0] * this_t_arg->size[1] * this_t_arg->size[2];
				size_in_bytes *= arg_metadata[2] - '0';	// 3 can't be specified as output so should be safe to do this
				size_in_bytes *= getChannelWidth(arg_metadata[1]);
				at->max_out_size = (at->max_out_size >= size_in_bytes) ? at->max_out_size : size_in_bytes;
			}

			this_t_arg->arg = createImageBuffer(context, this_s_arg->is_host_readable, this_s_arg->is_array, &(this_t_arg->format), this_t_arg->size);

			clErr = clSetKernelArg(kernel, j, sizeof(cl_mem), &(this_t_arg->arg));
			handleClError(clErr, "\nclSetKernelArg");

			at->args_cnt++;
		}
		// read-only and r/w args that don't create new mem objects need to be checked that they match
		else// if(arg_access == CL_KERNEL_ARG_ACCESS_READ_ONLY || (arg_access == CL_KERNEL_ARG_ACCESS_READ_WRITE && arg_metadata[3] != 'n'))
		{
			if(isValid)
				verifyReadArgTypeMatch(ref_arg->format, arg_metadata);
			else
				fputs("\nWarning: invalid argument metadata on argument to be read, can't provide type warnings\n", stderr);
		
			clErr = clSetKernelArg(kernel, j, sizeof(cl_mem), &(ref_arg->arg));
			handleClError(clErr, "\nclSetKernelArg");
		}
	}
}

// staging is a NULL terminated QStaging* array that is no longer than the kernels array
int prepQStages(cl_context context, const QStaging** staging, const cl_kernel* ref_kernels, QStage* stages, int max_stages, ArgTracker* at)
{
	cl_int clErr;
	
	int i=0;
	for(; staging[i]; ++i)
	{
		if(i==max_stages)
		{
			fputs("\nWarning: more stages in staging than max_stages\n", stderr);
			break;
		}

		int kern_idx = staging[i]->kernel_idx;

		// set the name in the current stage and print it, done early to easier identify errors
		clErr = clGetKernelInfo(ref_kernels[kern_idx], CL_KERNEL_FUNCTION_NAME, sizeof(stages[i].name), &(stages[i].name), NULL);
		handleClError(clErr, "clGetKernelInfo");
		printf("\nStaging kernel: %s ", stages[i].name);
		
		// it's maybe a little wasteful to always clone the reference kernel every time,
		// but at least I don't need to manually track if it's been used already this way
		cl_kernel curr_kern = clCloneKernel(ref_kernels[kern_idx], &clErr);
		handleClError(clErr, "clCloneKernel");
		stages[i].kernel = curr_kern;

		setKernelArgs(context, staging[i], curr_kern, at);

		TrackedArg* ref_arg = getRefArg(at, staging[i]->rel_ref);

		calcSizeByMode(ref_arg->size, &staging[i]->range, stages[i].range);
	}

	return i;
}


// creates a single channel cl_mem image from a file and attaches it to the tracked arg pointer provided
// the tracked arg must have the format pre-populated with a suitable way to interpret the raw image data
void imageFromFile(cl_context context, const char* fname, TrackedArg* tracked)
{
	int channels = getChannelCount(tracked->format.image_channel_order);
	
	// Read pixel data
	int dims[3];
	// the loading of the image hides a malloc deep in it
	unsigned char* data = stbi_load(fname, &dims[0], &dims[1], &dims[2], channels);
	if(!data)
	{
		perror("\nCouldn't open input image");
		exit(1);
	}

	printf("loaded %s, %i*%i image with %i channel(s), using %i channel(s).\n", fname, dims[0], dims[1], dims[2], channels);
	tracked->size[0] = dims[0];
	tracked->size[1] = dims[1];
	tracked->size[2] = 1;

	cl_int clErr;

	cl_image_desc image_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = tracked->size[0],
		.image_height = tracked->size[1],
		.image_depth = tracked->size[2],
		.image_array_size = 1,
		.image_row_pitch = 0,
		.image_slice_pitch = 0,
		.num_mip_levels = 0,
		.num_samples = 0,
		.buffer= NULL
	};

	// Create the input image object from the image file data
	tracked->arg = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_WRITE_ONLY, &(tracked->format), &image_desc, data, &clErr);
	handleClError(clErr, "clCreateImage");

	free(data);
}

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

	printf("\nCreating %zu*%zu*%zu buffer.\n", img_size[0], img_size[1], img_size[2]);
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
