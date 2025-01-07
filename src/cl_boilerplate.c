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
int getStringIndex(char const** list, char const* str)
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

/*/ helper function that returns the first position that a string differs by,
// assumes at least one is null terminated
int strDiffPos(char const* lhs, char const* rhs)
{
	int i = 0;
	while(lhs[i] == rhs[i])
	{
		if(lhs[i++] == '\0')	// if the matched character was a null character, stop comparing
			break;
	}
	return i;
}
*/

// Initializes a StagedQ object's arrays and counts
void allocStagedQArrays(QStaging const* staging, StagedQ* staged, clbp_Error* e)
{
	staged->img_arg_cnt = staging->img_arg_cnt;
	staged->stage_cnt = staging->stage_cnt;

	staged->img_args = malloc(staging->img_arg_cnt * sizeof(cl_mem));
	if(!staged->img_args)
	{
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "img_args array"};
		return;
	}

	staged->img_sizes = malloc(staging->img_arg_cnt * sizeof(Size3D));
	if(!staged->img_sizes)
	{
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "img_sizes array"};
		return;
	}

	staged->kernels = malloc(staging->kernel_cnt * sizeof(cl_kernel));
	if(!staged->kernels)
	{
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "cl_program array"};
		return;
	}
}

// applies the relative calculations for all arg sizes starting from the first non-hardcoded input argument
void calcRanges(QStaging const* staging, StagedQ* staged, clbp_Error* e)
{
	RangeData const* curr_range;
	Size3D const* ref_size;
	Size3D* sizes;
	
	sizes = staged->img_sizes;
	for(int i = staging->input_img_cnt; i < staged->img_arg_cnt; ++i)
	{
		curr_range = &staging->img_arg_stg[i].size;
		ref_size = &sizes[curr_range->ref_idx];
		e->err_code = calcSizeByMode(ref_size, curr_range, &sizes[i]);
		if(e->err_code)
		{
			e->detail = i;
			return;
		}
	}

	sizes = staged->ranges;
	for(int i = 0; i < staged->stage_cnt; ++i)
	{
		curr_range = &staging->kern_stg[i].range;
		ref_size = &sizes[curr_range->ref_idx];
		e->err_code = calcSizeByMode(ref_size, curr_range, &sizes[i]);
		if(e->err_code)
		{
			e->detail = -i;
			return;
		}
	}
}

// handles using staging data to selectively open kernel program source files and compile and link them into a single program binary
//TODO: add support for using pre-calculated ranges as defined constants
cl_program buildKernelProgsFromSource(cl_context context, cl_device_id device, const char* src_dir, QStaging* staging, const char* args, clbp_Error* e)
{
	assert(src_dir && staging && e);
	char fpath[1024];
	//TODO: add whole program binary caching by checking existence of compiled + linked bin,
	// and last modified dates match cached version for all sources in list
	cl_program* kprogs = malloc(staging->kernel_cnt * sizeof(cl_program));
	if(!kprogs)
	{
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "cl_program array"};
		return NULL;
	}

	// Read kernel program source file and place content into buffer
	for(int i = 0; i < staging->kernel_cnt; ++i)
	{
		//TODO: add binary caching/loading, needs to check existence of binary and last modified timestamp of source
		//append src dir to name and attempt read, unfortunately not smart enough to know about header changes but it'll have to do
		snprintf(fpath, sizeof(fpath)-1, "%s%s.cl", src_dir, staging->kprog_names[i]);
		char* k_src = readFileToCstring(fpath, e);
		if(e->err_code)
		{
			free(kprogs);
			return NULL;
		}

		// Create program from file
		kprogs[i] = clCreateProgramWithSource(context, 1, (const char**)&k_src, NULL, &e->err_code);
		free(k_src);
		if(e->err_code)
		{
			free(kprogs);
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
			free(kprogs);
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
			free(kprogs);
			if(e->err_code == CL_LINK_PROGRAM_FAILURE)
				handleClBuildProgram(e->err_code, linked_prog, device);
			e->detail = "clLinkProgram";
			return NULL;
	}
	//TODO: add release of individual kernel programs
	return linked_prog;
}

// creates actual kernel instances from staging data and stores it in the staged queue
void instantiateKernels(cl_context context, QStaging const* staging, const cl_program kprog, StagedQ* staged, clbp_Error* e)
{
	assert(staging && kprog && staged && e);
	for(int i = 0; i < staged->stage_cnt; ++i)
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

		staged->kernels[i] = kernel;
	}

	return;
}

// infers the access qualifiers of the image args as well as verifies that type data specified matches what the kernels expect of it
// meant to be run once after kernels have been instantiated for at least 1 staged queue, additional staged queues don't
// require re-runs of inferArgAccessAndVerifyFormats() since data extracted from the kernel instance args shouldn't change
void inferArgAccessAndVerifyFormats(QStaging* staging, StagedQ const* staged)
{
	printf("[Verifying kernel args]\n");
	// for each stage
	for(int i = 0; i < staged->stage_cnt; ++i)
	{
		cl_kernel curr_kern = staged->kernels[i];
		char const* kprog_name = staging->kprog_names[staging->kern_stg[i].kernel_idx];
		cl_uint arg_cnt;
		cl_uint err;
		printf("%i:	%s\n", i, kprog_name);
		err = clGetKernelInfo(curr_kern, CL_KERNEL_NUM_ARGS, sizeof(arg_cnt), &arg_cnt, NULL);
		staging->kern_stg[i].arg_cnt = arg_cnt;
		if(err)
		{
			handleClError(err, "clGetKernelInfo");
			perror("WARNING: couldn't get CL_KERNEL_NUM_ARGS\n"
			"	Skipping argument access qualifier inferencing and format verification.\n");
			continue;
		}
		// for each argument of the current kernel
		for(int j = 0; j < arg_cnt; ++j)
		{
			int arg_idx = staging->kern_stg[i].arg_idxs[j];
			char const* arg_name = staging->arg_names[arg_idx];
			printf("	%i:	%s", j, arg_name);
			ArgStaging* curr_arg = &staging->img_arg_stg[arg_idx];
			cl_kernel_arg_access_qualifier access_qual;
			err = clGetKernelArgInfo(curr_kern, j, CL_KERNEL_ARG_ACCESS_QUALIFIER, sizeof(access_qual), &access_qual, NULL);
			if(err)
			{
				handleClError(err, "clGetKernelArgInfo");
				perror("WARNING: couldn't get CL_KERNEL_ARG_ACCESS_QUALIFIER\n"
				"	Skipping argument access qualifier inferencing.\n");
			}
			else
			{
				cl_mem_flags* curr_flags = &curr_arg->flags;
				switch(access_qual)
				{
				//FIXME: not sure if this is correct, it might be that arguments that get written and read by different kernels
				// need to be read/write instead of read only + write only, once I know for sure, I'll fix this or remove this note
				case CL_KERNEL_ARG_ACCESS_READ_ONLY:
					// check for read before write, if none of these flags are set, nothing* could have written to it before this read occured
					// *except writing to it from the same kernel but that's undefined behavior and not portable and harder to check so I'm not checking that
					if(!(*curr_flags & (CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_WRITE_ONLY)))
						perror("WARNING: reading arg before writing to it.\n");
					*curr_flags |= CL_MEM_READ_ONLY;
					break;
				case CL_KERNEL_ARG_ACCESS_WRITE_ONLY:
					*curr_flags |= CL_MEM_WRITE_ONLY;
					break;
				case CL_KERNEL_ARG_ACCESS_READ_WRITE:
					*curr_flags |= CL_MEM_READ_WRITE;
					break;
			//	case CL_KERNEL_ARG_ACCESS_NONE:	//not an image or pipe, access qualifier doesn't apply
				default:
					perror("WARNING: non-image arg requested\n"
					"	Currently no non-image support implemented.\n");
					continue;	//TODO: currently doesn't handle non-image types, this is just placeholder code that would probably break if executed
				}
			}

			char arg_metadata[64];	// although only 4 entries are needed, reading the name will fail if there's not enough room for the whole name
			err = clGetKernelArgInfo(curr_kern, i, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_metadata), arg_metadata, NULL);
			if(err)
			{
				handleClError(err, "clGetKernelArgInfo");
				perror("WARNING: couldn't get CL_KERNEL_ARG_TYPE_NAME\n"
				"	Couldn't verify arg type.\n");
				continue;
			}

			puts(arg_metadata);//TODO: remove this debugging line once you know what all the types actually return
			enum clMemType mem_type = getStringIndex(memTypes, arg_metadata) + CLBP_OFFSET_MEMTYPE;
			if(mem_type >= CLBP_INVALID_MEM_TYPE || mem_type < CLBP_OFFSET_MEMTYPE)
			{
				perror("WARNING: non-mem object arg requested\n"
				"	Currently non-mem objects not supported.\n");
				continue;
			}

			if(mem_type != curr_arg->type)
			{
				fprintf(stderr, "WARNING: argument type mismatch\n"
				"	provided %s, requested %s.\n", arg_metadata, memTypes[curr_arg->type - CLBP_OFFSET_MEMTYPE]);
				continue;
			}
			// else, no warning, verified!

			// I attach type data in a similar style to Hungarian notation to the names so that the expected type backing of
			// the image is stored with the kernel itself and can be interpreted here by querying the name.
			// [0] == [f/h/u/i] the read/write type used with it, using a non matched cl_channel_type can lead to Undefined Behavior
			// [1] == [for f/h: s/u/f, for u/i: c/s/i] hint for what range of values are expected, float:signed norm/unsigned norm/full range, int: char/short/integer
			// [2] == [1/2/3/4] hint for how many channels it expects
			err = clGetKernelArgInfo(curr_kern, i, CL_KERNEL_ARG_NAME, sizeof(arg_metadata), arg_metadata, NULL);
			if(err)
			{
				handleClError(err, "clGetKernelArgInfo");
				perror("WARNING: couldn't get CL_KERNEL_ARG_NAME\n"
				"	Skipping image format verification.\n");
				continue;
			}

			char isValid = isArgMetadataValid(arg_metadata);
			if(!isValid)
			{
				fprintf(stderr, "WARNING: invalid argument metadata: %s\n"
				"	Skipping image format verification.\n", arg_metadata);
				continue;
			}

			if(!isMatchingChannelType(arg_metadata, curr_arg->format.image_channel_data_type))
				fprintf(stderr, "WARNING: channel data type mismatch\n");	//TODO: add more info of the mismatch

			if(ChannelOrderDiff(arg_metadata[2], curr_arg->format.image_channel_order))
				fprintf(stderr, "WARNING: channel count mismatch\n");	//TODO: add more info of the mismatch
				//NOTE: may need to add special processing for 3 channel items since those aren't required to be supported by the OpenCL spec
		}
	}
}

// fills in the ArgTracker according to the arg staging data in staging,
// assumes the ArgTracker was allocated big enough not to overrun it and
// is pre-populated with the expected number of hard-coded input entries
// such that it may add the first new entry at input_img_cnt
// returns the max number of bytes needed for reading out of any of the host readable buffers
//FIXME: currently doesn't take into account input args being read out for the return value
size_t instantiateImgArgs(cl_context context, QStaging const* staging, StagedQ* staged, clbp_Error* e)
{
	size_t max_out_sz = 0;
	cl_mem* img_args = staged->img_args;
	for(int i = staging->input_img_cnt; i < staging->img_arg_cnt; ++i)
	{
		ArgStaging* curr_arg = &staging->img_arg_stg[i];
		uint16_t* size = staged->img_sizes[i].d;
		cl_image_desc desc = {
			.image_type = curr_arg->type,
			.image_width = size[0],
			.image_height = size[1],
			.image_depth = size[2],
			.image_array_size = 1,
			.image_row_pitch = 0,
			.image_slice_pitch = 0,
			.num_mip_levels = 0,
			.num_samples = 0,
			.buffer = NULL
		};

		if(curr_arg->flags & (CL_MEM_HOST_READ_ONLY))
		{	// calculate output size

			size_t curr_size = getPixelSize(curr_arg->format);
			curr_size *= (size_t)size[0] * size[1] * size[2];
			if(max_out_sz < curr_size)
				max_out_sz = curr_size;
		}
		else// if(!(curr_arg->flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR))) //I suspect these flags might qualify too
			curr_arg->flags |= CL_MEM_HOST_NO_ACCESS;

		img_args[i] = clCreateImage(context, curr_arg->flags, &curr_arg->format, &desc, NULL, &e->err_code);
		if(e->err_code)
			e->detail = "clCreateImage";
		
		return max_out_sz;
	}
}

//TODO: add support for returning a list of host readable buffers
void setKernelArgs(cl_context context, QStaging const* staging, StagedQ* staged, clbp_Error* e)
{
	//for each stage
	for(int i = 0; i < staged->stage_cnt; ++i)
	{
		KernStaging const* curr_kstaging = &staging->kern_stg[i];
		cl_kernel curr_kern = staged->kernels[i];
		for(int j = 0; j < curr_kstaging->arg_cnt; ++j)
		{
			uint16_t arg_idx = curr_kstaging->arg_idxs[j];
			e->err_code = clSetKernelArg(curr_kern, j, sizeof(cl_mem), staged->img_args[arg_idx]);
			if(e->err_code)
			{
				fprintf(stderr, "@ stage %i (%s), arg %i (%s):", i, staging->kprog_names[curr_kstaging->kernel_idx], j, staging->arg_names[arg_idx]);
				e->detail = "clSetKernelArg";
				return;
			}
		}
	}
}

// creates a single channel cl_mem image from a file and attaches it to the tracked arg pointer provided
// the tracked arg must have the format pre-populated with a suitable way to interpret the raw image data
cl_mem imageFromFile(cl_context context, char const* fname, cl_image_format const* format, Size3D* size, clbp_Error* e)
{
	int channels = getChannelCount(format->image_channel_order);
	
	// Read pixel data
	int dims[3];
	// the loading of the image has a malloc deep in it
	uint8_t* data = stbi_load(fname, &dims[0], &dims[1], &dims[2], channels);
	if(!data)
	{
		*e = (clbp_Error){.err_code = CLBP_FILE_NOT_FOUND, .detail = "\nCouldn't open input image"};
		return NULL;
	}
	
	size->d[0] = dims[0];
	size->d[1] = dims[1];
	size->d[2] = 1;

	printf("loaded %s, %i*%i image with %i channel(s), using %i channel(s).\n", fname, dims[0], dims[1], dims[2], channels);

	cl_image_desc image_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = dims[0],
		.image_height = dims[1],
		.image_depth = 1,
		.image_array_size = 1,
		.image_row_pitch = 0,
		.image_slice_pitch = 0,
		.num_mip_levels = 0,
		.num_samples = 0,
		.buffer = NULL
	};

	cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;

	// Create the input image object from the image file data
	cl_mem img = clCreateImage(context, flags, format, &image_desc, data, &e->err_code);
	free(data);
	if(e->err_code)
		e->detail = "clCreateImage";

	return img;
}

// converts format of data to char array compatible read,
// data must point to a 32-bit aligned array. if it was malloc'd, it is aligned
// returns channel count since it's often needed after this and is already called here
uint8_t readImageAsCharArr(char* data, StagedQ* staged, uint16_t idx)
{
	uint8_t channel_cnt = getChannelCount(staged->format.image_channel_order);
	size_t length = arg->size[0] * arg->size[1] * arg->size[2] * channel_cnt;
	switch (arg->format.image_channel_data_type)
	{
	case CL_UNORM_INT8:
	case CL_UNSIGNED_INT8:
	case CL_SNORM_INT8:
	case CL_SIGNED_INT8:
		break;	// no change, data is already 1 byte per channel
	case CL_UNORM_INT16:
	case CL_UNSIGNED_INT16:
	case CL_SNORM_INT16:
	case CL_SIGNED_INT16:
	case CL_HALF_FLOAT:
		for(size_t i = 0; i < length; ++i)
			data[i] = ((int16_t*)data)[i] >> 8;	// assume msb is most important to preserve
		break;
	case CL_SIGNED_INT32:
	case CL_UNSIGNED_INT32:
	case CL_FLOAT:
		for(size_t i = 0; i < length; ++i)
			data[i] = ((int32_t*)data)[i] >> 24;	// assume msb is most important to preserve
		break;
	case CL_UNORM_SHORT_565:

	case CL_UNORM_SHORT_555:
	case CL_UNORM_INT_101010:
	case CL_UNORM_INT_101010_2:
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

void freeQStagingArrays(QStaging* staging)
{
	for(int i = 0; i < staging->stage_cnt; ++i)
	{
		free(staging->kern_stg[i].arg_idxs);
		//TODO: if you add kprog_names copying the names would need to be freed here
	}
	free(staging->kern_stg);
	free(staging->kprog_names);
	//TODO: if you add arg_names copying the names would need to be freed here
	free(staging->arg_names);
	free(staging->img_arg_stg);
}

void freeStagedQArrays(StagedQ* staged)
{
	free(staged->ranges);
	free(staged->img_sizes);
	//TODO: see if these items can be released earlier so that they all automatically get fully released when the
	// command queue gets released
	cl_uint err;
	for(int i = 0; i < staged->img_arg_cnt; ++i)
	{
		err = clReleaseMemObject(staged->img_args[i]);
		handleClError(err, "clReleaseMemObject");
	}
	free(staged->img_args);

	for(int i = 0; i < staged->stage_cnt; ++i)
	{
		err = clReleaseKernel(staged->kernels[i]);
		handleClError(err, "clReleaseKernel");
	}
	free(staged->kernels);
}