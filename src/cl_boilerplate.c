#include "cl_boilerplate.h"
#include "clbp_utils.h"
#include "cl_error_handlers.h"
#include "stb_image.h"

// attempts to get the first available GPU or if none available CPU
//TODO: actually implement multiple attempts to find a GPU, currently just takes the first device of the first platform
cl_device_id getPreferredDevice()
{
	cl_platform_id platform[4];
	cl_device_id device;
	cl_int clErr;

	// use the first platform
	clErr = clGetPlatformIDs(4, platform, NULL);
	handleClError(clErr, "clGetPlatformIDs");

	// use the first device
	clErr = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	handleClError(clErr, "clGetDeviceIDs");

	return device;
}

// adds the char* to the char* array if the contents are unique, the char* array
// MUST have unused entries filled with null pointers with an additional null
// pointer at list[max_entries]
// returns -1 if out of entries, or position if it either found it or added it
//NOTE: does not check str for a null pointer
int addUniqueString(char** list, int max_entries, char* str)
{
	int i;
	for(i = 0; (i < max_entries) && list[i]; ++i)
	{
		// if there's an exact match, it's already in the list and we can return early.
		if(!strncmp(list[i], str, strlen(str)))
			return i;
	}
	
	if(i >= max_entries)
		return -1;
	
	list[i] = str;
	return i;
}


// Searches through a string array (char** list) for a match to the contents of char* str
// Stops searching if it reaches a null pointer in the list. Returns -1 if no match is found.
//NOTE: does not check str for a null pointer
int getStringIndex(char** list, const char* str)
{
	int i = 0;
	while(list[i])
	{
		// if there's an exact match, it's in the list and we can return its index
		if(!strncmp(list[i], str, strlen(str)))
			return i;

		++i;
	}
	
	return -1;
}

cl_program buildKernelProgsFromSource(cl_context context, cl_device_id device, const char* src_dir, QStaging* staging, const char* args, cl_program* kprogs, clbp_Error* e)
{
	assert(src_dir && staging && kprogs && e);
	char fpath[1024];
	//TODO: add whole program binary caching by checking existence of compiled + linked bin,
	// and last modified dates match cached version for all sources in list

	// Read kernel program source file and place content into buffer
//	cl_uint kernel_cnt = 0;
	for(cl_uint i = 0; i < staging->kernel_cnt; ++i)
	{
		//TODO: add binary caching/loading, needs to check existence of binary and last modified timestamp of source
		//append src dir to name and attempt read, unfortunately not smart enough to know about header changes but it'll have to do
		snprintf(fpath, sizeof(fpath)-1, "%s%s.cl", src_dir, staging->kprog_names[i]);
		char* k_src = readFileToCstring(fpath, e);
		if(e->err_code)
			return NULL;

		// Create program from file
		kprogs[i] = clCreateProgramWithSource(context, 1, (const char**)&k_src, NULL, &e->err_code);
		free(k_src);
		if(e->err_code)
		{
			e->detail = "clCreateProgramWithSource";
			return NULL;
		}

		// Compile program
		// device should be singular and specified or else you can end up with multiple to a context,
		// error out, and then fail to print the log for the one that actually had the error
		printf("Compiling %s\n", fpath);
		e->err_code = clCompileProgram(kprogs[i], 1, &device, args, 0, NULL, NULL, NULL, NULL);
		if(e->err_code)
		{
			if(e->err_code == CL_COMPILE_PROGRAM_FAILURE)
				handleClBuildProgram(e->err_code, kprogs[i], device);
			e->detail = "clCompileProgram";
			return NULL;
		}
	}

	puts("Linking...\n");
	cl_program linked_prog = clLinkProgram(context, 1, &device, args, staging->kernel_cnt, kprogs, NULL, NULL, &e->err_code);
	if(e->err_code)
	{
			if(e->err_code == CL_LINK_PROGRAM_FAILURE)
				handleClBuildProgram(e->err_code, linked_prog, device);
			e->detail = "clLinkProgram";
			return NULL;
	}
	//TODO: add release of individual kernel programs
	return linked_prog;
}

// returns the max number of bytes needed for reading out of any of the host readable buffers
//TODO: add support for returning a list of host readable buffers
void setKernelArgs(cl_context context, const KernStaging* stage, cl_kernel kernel, ArgTracker* at, clbp_Error* e)
{
	cl_int clErr;
	// get the count of how many args the kernel has to iterate over
	cl_uint arg_cnt;
	clErr = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &arg_cnt, NULL);
	handleClError(clErr, "clGetKernelInfo");

	for(cl_uint j=0; j < arg_cnt; ++j)
	{
		printf("\n* [%i] ", j);
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
		ArgStaging* this_s_arg = &(stage->arg_idxs[j]);	//FIXME: broken here due to updates

		TrackedArg* ref_arg = &(at->args[this_s_arg->size.ref_idx]);

		char isValid = isArgMetadataValid(arg_metadata);

		// write-only and certain r/w arguments create new mem objects to write their outputs to
		//NOTE: theoretically you might have an uneeded output that you leave unassigned but I'm not supporting that, just rewrite it without it instead
		//TODO: add more diagnostic info to created args like channel count and type
		if(arg_access == CL_KERNEL_ARG_ACCESS_WRITE_ONLY || (arg_access == CL_KERNEL_ARG_ACCESS_READ_WRITE && arg_metadata[3] == 'n'))
		{
			if(!isValid)
			{
				fputs("Couldn't parse argument metadata, can't create image object", stderr);
				exit(1);
			}

			if(at->args_cnt >= at->max_args)
			{
				fputs("Not enough room in ArgTracker.args array", stderr);
				exit(1);
			}

			TrackedArg* this_t_arg = &(at->args[at->args_cnt]);

			this_t_arg->format.image_channel_data_type = getTypeFromMetadata(arg_metadata);
			this_t_arg->format.image_channel_order = getOrderFromMetadata(arg_metadata);
			calcSizeByMode(ref_arg->size, &this_s_arg->size, this_t_arg->size);

			// if this is a host readable output, we need to see if the size in bytes is bigger than any previous args so the
			// final read buffer can be allocated large enough
			if(this_s_arg->is_host_readable)
			{
				size_t size_in_bytes = this_t_arg->size[0] * this_t_arg->size[1] * this_t_arg->size[2];
				size_in_bytes *= arg_metadata[2] - '0';	// 3 can't be specified as output so should be safe to do this
				size_in_bytes *= getChannelWidth(arg_metadata[1]);
				at->max_out_size = (at->max_out_size >= size_in_bytes) ? at->max_out_size : size_in_bytes;
			}

//FIXME: no magic values vvv here vvv come back later and properly fix handling of the arg type when you understand the other types better
			this_t_arg->arg = createImageBuffer(context, this_s_arg->is_host_readable, this_s_arg->type == 'a', &(this_t_arg->format), this_t_arg->size);

			clErr = clSetKernelArg(kernel, j, sizeof(cl_mem), &(this_t_arg->arg));
			handleClError(clErr, "clSetKernelArg");

			at->args_cnt++;
		}
		// read-only and r/w args that don't create new mem objects need to be checked that they match
		else// if(arg_access == CL_KERNEL_ARG_ACCESS_READ_ONLY || (arg_access == CL_KERNEL_ARG_ACCESS_READ_WRITE && arg_metadata[3] != 'n'))
		{
			if(isValid)
				verifyReadArgTypeMatch(ref_arg->format, arg_metadata);
			else
				fputs("Warning: invalid argument metadata on argument to be read, can't provide type warnings\n", stderr);
		
			printf("Using %zu*%zu*%zu buffer with format %c%c%i.", ref_arg->size[0],ref_arg->size[1],ref_arg->size[2], \
			getDeviceRWType(ref_arg->format.image_channel_data_type), getArgStorageType(ref_arg->format.image_channel_data_type), getChannelCount(ref_arg->format.image_channel_order));
			clErr = clSetKernelArg(kernel, j, sizeof(cl_mem), &(ref_arg->arg));
			handleClError(clErr, "clSetKernelArg");
		}
	}
}

void prepQStages(cl_context context, const QStaging* staging, const cl_program kprog, QStage* stages, ArgTracker* at, clbp_Error* e)
{
	assert(staging && kprog && stages && at && e);
	for(int i = 0; i < staging->stage_cnt; ++i)
	{
		KernStaging* curr_stage = &staging->kern_stg[i];
		char* kprog_name = staging->kprog_names[curr_stage->kernel_idx];
		printf("Staging %s\n", kprog_name);
		cl_kernel kernel = clCreateKernel(kprog, kprog_name, &e->err_code);
		if(e->err_code)
		{
			e->detail = "clCreateKernel";
			return;
		}

		// set the name in the current stage and print it, done early to easier identify errors
	//	e->err_code = clGetKernelInfo(ref_kernels[kern_idx], CL_KERNEL_FUNCTION_NAME, sizeof(stages[i].name), &(stages[i].name), NULL);
	//	handleClError(clErr, "clGetKernelInfo");
	//	printf("\n\nStaging kernel: %s", stages[i].name);
		
		// it's maybe a little wasteful to always clone the reference kernel every time,
		// but at least I don't need to manually track if it's been used already this way
		/*	// clCloneKernel() not available in OpenCL1.2
		cl_kernel curr_kern = clCloneKernel(ref_kernels[kern_idx], &clErr);
		handleClError(clErr, "clCloneKernel");
		*/
		//FIXME: Temp fix for OpenCL 1.2 support, just assign the ref_kernel and don't free it, means each can only be used once
	//	cl_kernel curr_kern = ref_kernels[kern_idx];
		stages[i].kernel = kernel;
		setKernelArgs(context, curr_stage, kernel, at, e);

		TrackedArg* ref_arg = &(at->args[curr_stage->range.ref_idx]);

		calcSizeByMode(ref_arg->size, &curr_stage->range, stages[i].range);
	}

	return;
}


// creates a single channel cl_mem image from a file and attaches it to the tracked arg pointer provided
// the tracked arg must have the format pre-populated with a suitable way to interpret the raw image data
void imageFromFile(cl_context context, const char* fname, TrackedArg* tracked)
{
	int channels = getChannelCount(tracked->format.image_channel_order);
	
	// Read pixel data
	int dims[3];
	// the loading of the image has a malloc deep in it
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
		.buffer = NULL
	};

	// Create the input image object from the image file data
	tracked->arg = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_WRITE_ONLY, &(tracked->format), &image_desc, data, &clErr);
	handleClError(clErr, "clCreateImage");

	free(data);
}

// converts format of data to char array compatible read, returns channel count since it's often needed after this and is already called here
unsigned char readImageAsCharArr(char* data, TrackedArg* arg)
{
	unsigned char channel_cnt = getChannelCount(arg->format.image_channel_order);
	size_t length = arg->size[0] * arg->size[1] * arg->size[2] * channel_cnt;
	switch (arg->format.image_channel_data_type)
	{
//	case CL_UNORM_SHORT_565:
//	case CL_UNORM_SHORT_555:
	case CL_UNORM_INT8:
	case CL_UNSIGNED_INT8:
	case CL_SNORM_INT8:
	case CL_SIGNED_INT8:
		break;	// no change, data is already 1 byte per channel
//	case CL_UNORM_INT_101010:
//	case CL_UNORM_INT_101010_2:
	case CL_UNORM_INT16:
	case CL_UNSIGNED_INT16:
	case CL_SNORM_INT16:
	case CL_SIGNED_INT16:
		for(size_t i = 0; i < length; ++i)
			data[i] = ((int16_t*)data)[i] >> 8;	// assume 16-bit normalized so msb is most important to preserve

		break;
	case CL_SIGNED_INT32:
	case CL_UNSIGNED_INT32:
		for(size_t i = 0; i < length; ++i)
			data[i] = ((int32_t*)data)[i] >> 24;	// assume 32-bit normalized so msb is most important to preserve

		break;
	case CL_HALF_FLOAT:	//TODO: add support for float16
//		for(size_t i = 0; i < length; ++i)
//			data[i] = ((float*)data)[i] * 128 + 0.5;	// assume +/- 1.0 normalization

		break;
	case CL_FLOAT:
		for(size_t i = 0; i < length; ++i)
			data[i] = ((float*)data)[i] * 128 + 0.5;	// assume +/- 1.0 normalization

		break;
	}
	return channel_cnt;
}

// Returned pointer must be freed when done using
char* readFileToCstring(char* fname, clbp_Error* e)
{
	assert(fname && e);
//	printf("Reading \"%s\"\n", fname);

	FILE* k_src_handle = fopen(fname, "r");
	if(k_src_handle == NULL)
	{
		*e = (clbp_Error){.err_code = CLBP_FILE_NOT_FOUND, .detail = fname};
		return NULL;
	}
	// get rough file size and allocate string
	fseek(k_src_handle, 0, SEEK_END);
	long k_src_size = ftell(k_src_handle);
	rewind(k_src_handle);

	char* manifest = malloc(k_src_size + 1);	// +1 to have enough room to add null termination
	if(!manifest)
	{
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = fname};
		return NULL;
	}

	// contents may be smaller due to line endings being partially stripped on read
	k_src_size = fread(manifest, sizeof(char), k_src_size, k_src_handle);
	fclose(k_src_handle);
	// terminate the string properly
	manifest[k_src_size] = '\0';

	return manifest;
}