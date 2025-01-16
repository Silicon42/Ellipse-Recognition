#include "cl_boilerplate.h"
#include "clbp_utils.h"
#include "cl_error_handlers.h"
#include "stb_image.h"

#define CLBP_MEM_RW	(CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY)

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
		if(!strncmp(list[i], str, strlen(str)+1))
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
		if(!strncmp(list[i], str, strlen(str)+1))
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
cl_int allocStagedQArrays(QStaging const* staging, StagedQ* staged)
{
	staged->img_arg_cnt = staging->img_arg_cnt;
	staged->stage_cnt = staging->stage_cnt;

	size_t size3d_byte_cnt = (staged->img_arg_cnt + staged->stage_cnt) * sizeof(Size3D);
	staged->img_sizes = malloc(size3d_byte_cnt);
	staged->ranges = staged->img_sizes + staged->img_arg_cnt;

	size_t cl_ptr_byte_cnt = (staged->img_arg_cnt + staging->kernel_cnt) * sizeof(cl_mem);
	staged->img_args = malloc(cl_ptr_byte_cnt);
	staged->kernels = (cl_kernel*)staged->img_args + staged->img_arg_cnt;

	// check for failed allocation and free if it was partially allocated
	if(!staged->img_sizes || !staged->img_args)
	{
		free(staged->img_sizes);
		free(staged->img_args);
		return CLBP_OUT_OF_MEMORY;
	}
	return CLBP_OK;
}

// applies the relative calculations for all arg sizes starting from the first non-hardcoded input argument
void calcRanges(QStaging const* staging, StagedQ* staged, clbp_Error* e)
{
	Size3D* sizes = staged->img_sizes;

	printf("Calculating %i image argument sizes... ", staged->img_arg_cnt);
	calcSizeByMode(sizes, staging->arg_size_calcs, sizes, staged->img_arg_cnt, e);
	if(e->err_code)
		return;
	puts("Done.");

	printf("Calculating %i NDRanges... ", staged->stage_cnt);
	calcSizeByMode(sizes, staging->range_calcs, staged->ranges, staged->stage_cnt, e);
	if(e->err_code)
		return;
	puts("Done.");
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
	printf("Compiling %i kernel programs.\n", staging->kernel_cnt);
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
			if(e->err_code == CL_COMPILE_PROGRAM_FAILURE)
				handleClBuildProgram(e->err_code, kprogs[i], device);
			free(kprogs);
			e->detail = "clCompileProgram";
			return NULL;
		}
	}

	fputs("Linking... ", stdout);
	cl_program linked_prog = clLinkProgram(context, 1, &device, args, staging->kernel_cnt, kprogs, NULL, NULL, &e->err_code);
	if(e->err_code)
	{
			free(kprogs);
			if(e->err_code == CL_LINK_PROGRAM_FAILURE)
				handleClBuildProgram(e->err_code, linked_prog, device);
			e->detail = "clLinkProgram";
			return NULL;
	}
	puts("Done.");
	//TODO: add release of individual kernel programs
	return linked_prog;
}

// creates actual kernel instances from staging data and stores it in the staged queue
void instantiateKernels(QStaging const* staging, const cl_program kprog, StagedQ* staged, clbp_Error* e)
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
}

// infers the access qualifiers of the image args as well as verifies that type data specified matches what the kernels expect of it
// meant to be run once after kernels have been instantiated for at least 1 staged queue, additional staged queues don't
// require re-runs of inferArgAccessAndVerifyFormats() since data extracted from the kernel instance args shouldn't change
//TODO: many of the warnings would interupt the printing of the argument info, see what can be done to tidy up so that they print after the line is done
void inferArgAccessAndVerifyFormats(QStaging* staging, StagedQ const* staged)
{
	printf("[Verifying kernel args]");
	// for each stage
	for(int i = 0; i < staged->stage_cnt; ++i)
	{
		char is_last_stage = (i+1 == staged->stage_cnt);	// anything written by the last stage gets set to host readable
		cl_kernel curr_kern = staged->kernels[i];
		char const* kprog_name = staging->kprog_names[staging->kern_stg[i].kernel_idx];
		cl_uint arg_cnt;
		cl_uint err;
		printf("\n(%i) %s", i, kprog_name);
		err = clGetKernelInfo(curr_kern, CL_KERNEL_NUM_ARGS, sizeof(arg_cnt), &arg_cnt, NULL);
		staging->kern_stg[i].arg_cnt = arg_cnt;
		if(err)
		{
			handleClError(err, "clGetKernelInfo");
			fputs("\nWARNING: couldn't get CL_KERNEL_NUM_ARGS. Skipping argument access qualifier inferencing and format verification.", stderr);
			continue;
		}
		// for each argument of the current kernel
		for(cl_uint j = 0; j < arg_cnt; ++j)
		{
			int arg_idx = staging->kern_stg[i].arg_idxs[j];
			char const* arg_name = staging->arg_names[arg_idx];
			ArgStaging* curr_arg = &staging->img_arg_stg[arg_idx];
			printf("\n	[%i] (%s) %s of channel type %s x %i ",
				j, memTypes[curr_arg->type - CLBP_OFFSET_MEMTYPE], arg_name,
				channelTypes[curr_arg->format.image_channel_data_type - CLBP_OFFSET_CHANNEL_TYPE],
				getChannelCount(curr_arg->format.image_channel_order));
			cl_kernel_arg_access_qualifier access_qual;
			err = clGetKernelArgInfo(curr_kern, j, CL_KERNEL_ARG_ACCESS_QUALIFIER, sizeof(access_qual), &access_qual, NULL);
			if(err)
			{
				handleClError(err, "clGetKernelArgInfo");
				fputs("\nWARNING: couldn't get CL_KERNEL_ARG_ACCESS_QUALIFIER. Skipping argument access qualifier inferencing.", stderr);
			}
			else
			{
				cl_mem_flags* curr_flags = &curr_arg->flags;
				switch(access_qual)
				{
				// this can cause situations where mutually exclusive flags are set, however those get fixed in instantiateImgArgs()
				// right before clCreateImage() is called and is much simpler to reason about if there is only a read flag and a write
				// flag which can be checked at the end if they're both set and then be cleared
				case CL_KERNEL_ARG_ACCESS_READ_ONLY:
					// check for read before write, if none of these flags are set, nothing* could have written to it before this read occured
					// *except writing to it from the same kernel but that's undefined behavior and not portable and harder to check so I'm not checking that
					if(!(*curr_flags & (CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_WRITE_ONLY)))
						fputs("\nWARNING: reading arg before writing to it.", stderr);
					*curr_flags |= CL_MEM_READ_ONLY;
					break;
				case CL_KERNEL_ARG_ACCESS_READ_WRITE:
					*curr_flags |= CL_MEM_READ_ONLY;
				case CL_KERNEL_ARG_ACCESS_WRITE_ONLY:
					*curr_flags |= CL_MEM_WRITE_ONLY;
					if(is_last_stage)
						*curr_flags |= CL_MEM_HOST_READ_ONLY;
					break;
			//	case CL_KERNEL_ARG_ACCESS_NONE:	//not an image or pipe, access qualifier doesn't apply
				default:
					fputs("\nWARNING: non-image arg requested. Currently no non-image support implemented.", stderr);
					continue;	//TODO: currently doesn't handle non-image types, this is just placeholder code that would probably break if executed
				}
			}

			char arg_metadata[64];	// although only 4 entries are needed, reading the name will fail if there's not enough room for the whole name
			err = clGetKernelArgInfo(curr_kern, j, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_metadata), arg_metadata, NULL);
			if(err)
			{
				handleClError(err, "clGetKernelArgInfo");
				fputs("\nWARNING: couldn't get CL_KERNEL_ARG_TYPE_NAME. Couldn't verify arg type.", stderr);
				continue;
			}

			printf("-> (%s) ", arg_metadata);
			enum clMemType mem_type = getStringIndex(memTypes, arg_metadata) + CLBP_OFFSET_MEMTYPE;

			// I attach type data in a similar style to Hungarian notation to the names so that the expected type backing of
			// the image is stored with the kernel itself and can be interpreted here by querying the name.
			// [0] == [f/h/u/i] the read/write type used with it, using a non matched cl_channel_type can lead to Undefined Behavior
			// [1] == [for f/h: s/u/f, for u/i: c/s/i] hint for what range of values are expected, float:signed norm/unsigned norm/full range, int: char/short/integer
			// [2] == [1/2/3/4] hint for how many channels it expects
			err = clGetKernelArgInfo(curr_kern, j, CL_KERNEL_ARG_NAME, sizeof(arg_metadata), arg_metadata, NULL);
			if(err)
			{
				handleClError(err, "clGetKernelArgInfo");
				fputs("\nWARNING: couldn't get CL_KERNEL_ARG_NAME Skipping image format verification.", stderr);
				continue;
			}
			printf("%s ", arg_metadata);

			if(mem_type >= CLBP_INVALID_MEM_TYPE || mem_type < CLBP_OFFSET_MEMTYPE)
			{
				fputs("\nWARNING: non-mem object arg requested Currently non-mem objects not supported.", stderr);
				continue;
			}

			if(mem_type != curr_arg->type)
			{
				fputs("\nWARNING: argument type mismatch.", stderr);
				continue;
			}
			// else, no warning, verified!
			char isValid = isArgMetadataValid(arg_metadata);
			if(!isValid)
			{
				fputs("\nWARNING: invalid argument metadata. Skipping image format verification.", stderr);
				continue;
			}

			if(!isMatchingChannelType(arg_metadata, curr_arg->format.image_channel_data_type))
				fputs("\nWARNING: channel data type mismatch", stderr);

			if(ChannelOrderDiff(arg_metadata[2], curr_arg->format.image_channel_order) < 0)
				fputs("\nWARNING: channel count mismatch", stderr);	//TODO: add more info of the mismatch
				//NOTE: may need to add special processing for 3 channel items since those aren't required to be supported by the OpenCL spec
		}
	}
	putchar('\n');
}

// fills in the ArgTracker according to the arg staging data in staging,
// assumes the ArgTracker was allocated big enough not to overrun it and
// is pre-populated with the expected number of hard-coded input entries
// such that it may add the first new entry at input_img_cnt
// returns the max number of bytes needed for reading out of any of the host readable buffers
size_t instantiateImgArgs(cl_context context, QStaging const* staging, StagedQ* staged, clbp_Error* e)
{
	size_t max_out_sz = 0;
	cl_mem* img_args = staged->img_args;
	for(int i = 0; i < staging->img_arg_cnt; ++i)
	{
		ArgStaging* curr_arg = &staging->img_arg_stg[i];
		cl_mem_flags flags = curr_arg->flags;
		size_t* size = staged->img_sizes[i].d;
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

		// read only and write only flags are mutually exclusive so if they both occur,
		// clear them to go back to default read/write behavior
		if((flags & CLBP_MEM_RW) == CLBP_MEM_RW)
			flags ^= CLBP_MEM_RW;

		if(flags & (CL_MEM_HOST_READ_ONLY))
		{	// calculate output size
			size_t curr_size = getPixelSize(curr_arg->format);
			curr_size *= (size_t)size[0] * size[1] * size[2];
			if(max_out_sz < curr_size)
				max_out_sz = curr_size;
		}
		else// if(!(curr_arg->flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR))) //I suspect these flags might qualify too
			flags |= CL_MEM_HOST_NO_ACCESS;

		img_args[i] = clCreateImage(context, flags, &curr_arg->format, &desc, (i < staging->input_img_cnt) ? staging->input_imgs[i] : NULL, &e->err_code);
		if(e->err_code)
		{
			e->detail = "clCreateImage";
			return 0;
		}
	}
	return max_out_sz;
}

//TODO: add support for returning a list of host readable buffers
void setKernelArgs(QStaging const* staging, StagedQ* staged, clbp_Error* e)
{
	//for each stage
	for(int i = 0; i < staged->stage_cnt; ++i)
	{
		KernStaging const* curr_kstaging = &staging->kern_stg[i];
		cl_kernel curr_kern = staged->kernels[i];
		for(int j = 0; j < curr_kstaging->arg_cnt; ++j)
		{
			uint16_t arg_idx = curr_kstaging->arg_idxs[j];
			e->err_code = clSetKernelArg(curr_kern, j, sizeof(cl_mem), &staged->img_args[arg_idx]);
			if(e->err_code)
			{
				fprintf(stderr, "@ stage %i (%s), arg %i (%s): ",
					i, staging->kprog_names[curr_kstaging->kernel_idx], j, staging->arg_names[arg_idx]);
				e->detail = "clSetKernelArg";
				return;
			}
		}
	}
}

// reads an image from file with the requested number of channels and attaches the data to the staging object
// must have the format and type pre-populated with a suitable way to interpret the raw image data
void inputImagesFromFiles(char const** fnames, QStaging* staging, clbp_Error* e)
{
	ArgStaging* curr_img;
	int x, y, ch;

	for(int i = 0; i < staging->input_img_cnt; ++i)
	{
		curr_img = &staging->img_arg_stg[i];
		int channels = getChannelCount(curr_img->format.image_channel_order);
		
		// Read pixel data
		// the loading of the image has a malloc deep in it
		staging->input_imgs[i] = stbi_load(fnames[i], &x, &y, &ch, channels);
		if(!staging->input_imgs[i])
		{
			*e = (clbp_Error){.err_code = CLBP_FILE_NOT_FOUND, .detail = "\nCouldn't open input image"};
			return;
		}

		staging->arg_size_calcs[i] = (RangeData){.param = {x,y,1}, .mode = CLBP_RM_EXACT, .ref_idx = 0};

		printf("loaded %s, %i*%i image with %i channel(s), using %i channel(s).\n", fnames[i], x, y, ch, channels);

		curr_img->flags = CL_MEM_COPY_HOST_PTR;
	}
}

// converts format of data to char array compatible read,
// data must point to a 32-bit aligned array. if it was malloc'd, it is aligned
// returns channel count since it's often needed after this and is already called here
uint8_t readImageAsCharArr(char* data, StagedQ const* staged, uint16_t idx)
{
	cl_image_format format;
	cl_uint err = clGetImageInfo(staged->img_args[idx], CL_IMAGE_FORMAT, sizeof(format), &format, NULL);
	if(err)	// Should never happen but just in case
	{
		handleClError(err, "clGetImageInfo->CL_IMAGE_FORMAT");
		perror("How did you get here?");
		return 0;
	}

	uint8_t channel_cnt = getChannelCount(format.image_channel_order);
	size_t const* size = staged->img_sizes[idx].d;
	size_t pix_len = size[0] * size[1] * size[2];
	// the packed channel types don't have their processing length multiplied by the channel count
	// instead they process on a per pixel basis since their channel widths aren't uniform/cross byte bounds
	size_t ch_len = pix_len * channel_cnt;
	uint32_t pixel;
	
	switch(format.image_channel_data_type)
	{
	case CLBP_UNORM_INT8:
	case CLBP_UNSIGNED_INT8:
	case CLBP_SNORM_INT8:
	case CLBP_SIGNED_INT8:
		break;	// no change, data is already 1 byte per channel
	case CLBP_UNORM_INT16:
	case CLBP_UNSIGNED_INT16:
	case CLBP_SNORM_INT16:
	case CLBP_SIGNED_INT16:
	case CLBP_HALF_FLOAT:
		for(size_t i = 0; i < ch_len; ++i)
			data[i] = ((int16_t*)data)[i] >> 8;		// assume msb is most important to preserve
		break;
	case CLBP_UNSIGNED_INT32:
	case CLBP_SIGNED_INT32:
	case CLBP_FLOAT:
		for(size_t i = 0; i < ch_len; ++i)
			data[i] = ((int32_t*)data)[i] >> 24;	// assume msb is most important to preserve
		break;
	case CLBP_UNORM_SHORT_565:
		// since conversion will expand the array, must work backward to avoid overwriting data
		for(size_t j = ch_len, i = pix_len - 1; i < pix_len; --i)
		{
			pixel = ((uint16_t*)data)[i];
			data[--j] = pixel << 3;				//blue
			data[--j] = (pixel >> 3) & 0xFC;	//green
			data[--j] = (pixel >> 8) & 0xF8;	//red
		}
		break;
	case CLBP_UNORM_SHORT_555:
		// since conversion will expand the array, must work backward to avoid overwriting data
		for(size_t j = ch_len, i = pix_len - 1; i < pix_len; --i)
		{
			pixel = ((uint16_t*)data)[i];
			if(channel_cnt == 4)
				data[--j] = (pixel >> 8) | 0x7F;	//full alpha if set, half if unset
			data[--j] = pixel << 3;				//blue
			data[--j] = (pixel >> 2) & 0xF8;	//green
			data[--j] = (pixel >> 7) & 0xF8;	//red
		}
		break;
	case CLBP_UNORM_INT_101010:
		for(size_t j = 0, i = 0; i < pix_len; ++i)
		{
			pixel = ((uint32_t*)data)[i];
			data[j++] = pixel >> 22;	//red
			data[j++] = pixel >> 12;	//green
			data[j++] = pixel >> 2;		//blue
			if(channel_cnt == 4)
				data[j++] = (pixel >> 24) | 0x6F;	//full alpha if both set, 1/4th if both unset
		}
		break;
	case CLBP_UNORM_INT_101010_2:
		for(size_t j = 0, i = 0; i < pix_len; ++i)
		{
			pixel = ((uint32_t*)data)[i];
			data[j++] = pixel >> 24;	//red
			data[j++] = pixel >> 14;	//green
			data[j++] = pixel >> 4;		//blue
			data[j++] = (pixel << 6) | 0x6F;	//full alpha if both set, 1/4th if both unset
		}
		break;
	}
	//signed types need to be offset to be semi-human understandable visually
	if(isChannelTypeSigned(format.image_channel_data_type))
	{
		if(format.image_channel_data_type == CL_FLOAT || format.image_channel_data_type == CL_HALF_FLOAT)
		{	// true floats need extra processing to convert the biased exponent to logical order
			for(size_t i = 0; i < ch_len; ++i)
				data[i] = (data[i] < 0) ? -data[i] : data[i] | 0x80;
		}
		else
		{
			for(size_t i = 0; i < ch_len; ++i)
				data[i] += 128;	//toggle top bit
		}
	}
	return channel_cnt;
}

// Returned pointer must be freed when done using, no need to free on error
char* readFileToCstring(char* fname, clbp_Error* e)
{
	assert(fname && e);
//	printf("Reading \"%s\"\n", fname);

	FILE* file = fopen(fname, "r");
	if(file == NULL)
	{
		*e = (clbp_Error){.err_code = CLBP_FILE_NOT_FOUND, .detail = fname};
		return NULL;
	}
	// get rough file size and allocate string
	fseek(file, 0, SEEK_END);
	long f_size = ftell(file);
	rewind(file);

	char* contents = malloc(f_size + 1);	// +1 to have enough room to add null termination
	if(!contents)
	{
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = fname};
		return NULL;
	}

	// contents may be smaller due to line endings being partially stripped on read
	f_size = fread(contents, sizeof(char), f_size, file);
	fclose(file);
	// terminate the string properly
	contents[f_size] = '\0';

	return contents;
}

void freeQStagingArrays(QStaging* staging)
{
	for(int i = 0; i < staging->input_img_cnt; ++i)
	{
		free(staging->input_imgs[i]);
	}
	free(staging->input_imgs);

	for(int i = 0; i < staging->stage_cnt; ++i)
	{
		free(staging->kern_stg[i].arg_idxs);
		//TODO: if you add kprog_names copying the names would need to be freed here
	}
	free(staging->kern_stg);
	free(staging->img_arg_stg);
	//TODO: if you add arg_names copying the names would need to be freed here
	free(staging->kprog_names);
}

void freeStagedQArrays(StagedQ* staged)
{
	free(staged->img_sizes);
	//TODO: see if these items can be released earlier so that they all automatically get fully released when the
	// command queue gets released
	cl_uint err;
	for(int i = 0; i < staged->img_arg_cnt; ++i)
	{
		err = clReleaseMemObject(staged->img_args[i]);
		handleClError(err, "clReleaseMemObject");
	}

	for(int i = 0; i < staged->stage_cnt; ++i)
	{
		err = clReleaseKernel(staged->kernels[i]);
		handleClError(err, "clReleaseKernel");
	}
	free(staged->img_args);
}