
#include "common_error_handlers.h"
#include "stb_image.h"


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
	clErr = clGetDeviceIDs(&platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	handleClError(clErr, "clGetDeviceIDs");

	return device;
}

cl_program buildProgramFromFile(cl_context context, cl_device_id device, const char* fname, const char* args)
{
	cl_program program;
	FILE *program_handle;
	char *program_log;
	size_t program_size, log_size;
	cl_int clErr;

	// Read program file and place content into buffer
	program_handle = fopen(fname, "r");
	if(program_handle == NULL)
	{
		perror("Couldn't find the kernel program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);	//TODO: see if there is a better way of getting program size than this
	program_size = ftell(program_handle);
	rewind(program_handle);

	char* program_buffer = (char*)malloc(program_size + 1);
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);
	program_buffer[program_size] = '\0';

	// Create program from file
	program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &program_size, &clErr);
	handleClError(clErr, "clCreateProgramWithSource");
	free(program_buffer);

	// Build program
	clErr = clBuildProgram(program, 1, &device, args, NULL, NULL);
	if(clErr)
	{
		// Find size of log and print to std output
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		handleClError(clErr, "clGetProgramBuildInfo");
		program_log = (char*) malloc(log_size+1);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, program_log, NULL);
		handleClError(clErr, "clGetProgramBuildInfo");
		program_log[log_size] = '\0';
		perror((const char*)program_log);
		free(program_log);
		exit(1);
	}

	return program;
}

// creates a single channel cl_mem image from a file
cl_mem imageFromFile(cl_context context, const char* fname, size_t* img_size)
{
	// Read pixel data
	int dims[3];
	unsigned char* data = stbi_load(fname, &dims[0], &dims[1], &dims[2], 1);
	if(!data)
	{
		perror("Couldn't open input image");
		exit(1);
	}

	img_size[0] = dims[0];
	img_size[1] = dims[1];
	img_size[2] = 1;

	cl_int clErr;

	cl_image_format img_format = {
		.image_channel_order = CL_R,
		.image_channel_data_type = CL_UNORM_INT8
	};

	cl_image_desc image_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = img_size[0],
		.image_height = img_size[1],
		.image_depth = 1,
		.image_array_size = 1,
		.image_row_pitch = 0,
		.image_slice_pitch = 0,
		.num_mip_levels = 0,
		.num_samples = 0,
		.buffer= NULL
	};

	// Create the input image object from the image file data
	cl_mem in_texture = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_WRITE_ONLY, &img_format, &image_desc, data, &clErr);
	handleClError(clErr, "in clCreateImage");

	return in_texture;
}

cl_mem imageOutputBuffer(cl_context context, const size_t* img_size)
{
	cl_int clErr;

	cl_image_format img_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8
	};

	cl_image_desc image_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = img_size[0],
		.image_height = img_size[1],
		.image_depth = 1,
		.image_array_size = 1,
		.image_row_pitch = 0,
		.image_slice_pitch = 0,
		.num_mip_levels = 0,
		.num_samples = 0,
		.buffer= NULL
	};

	cl_mem out_texture = clCreateImage(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, &img_format, &image_desc, NULL, &clErr);
	handleClError(clErr, "out clCreateImage");

	return out_texture;
}

cl_mem imageIntermediateBuffer(cl_context context, const size_t* img_size, cl_channel_order order)
{
	cl_int clErr;

	cl_image_format img_format = {
		.image_channel_order = order,
		.image_channel_data_type = CL_FLOAT
	};

	cl_image_desc image_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = img_size[0],
		.image_height = img_size[1],
		.image_depth = 1,
		.image_array_size = 1,
		.image_row_pitch = 0,
		.image_slice_pitch = 0,
		.num_mip_levels = 0,
		.num_samples = 0,
		.buffer= NULL
	};

	cl_mem inter_texture = clCreateImage(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, &img_format, &image_desc, NULL, &clErr);
	handleClError(clErr, "intermediate clCreateImage");

	return inter_texture;
}
