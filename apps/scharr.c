#include <stdio.h>
#include <CL/cl.h>
#include "common_error_handlers.h"
#include "stb_image.h"
#include "stb_image_write.h"

#define PROGRAM_NAME "scharr"
#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)
//#define SCALE_FACTOR 2


// attempts to get the first available GPU or if none available CPU
//TODO: actually implement multiple attempts to find a GPU, currently just takes the first device of the first platform
//TODO: should be moved to a separate file for helper functions
cl_device_id getPreferredDevice()
{
	cl_platform_id platform[2];
	cl_device_id device;
	cl_int clErr;

	// use the first platform
	clErr = clGetPlatformIDs(2, platform, NULL);
	handleClError(clErr, "clGetPlatformIDs");

	// use the first device
	clErr = clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_ALL, 1, &device, NULL);
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
	cl_int clErr;

	cl_image_format img_format = {
		.image_channel_order = CL_R,
		.image_channel_data_type = CL_UNORM_INT8
	};

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
		.buffer= NULL
	};

	// Create the input image object from the PNG data
	cl_mem in_texture = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &img_format, &image_desc, data, &clErr);
	handleClError(clErr, "in clCreateImage");

	return in_texture;
}

cl_mem imageOutputBuffer(cl_context context, size_t width, size_t height)
{
	cl_int clErr;

	cl_image_format img_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8
	};

	cl_image_desc image_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = width,
		.image_height = height,
		.image_depth = 1,
		.image_array_size = 1,
		.image_row_pitch = 0,
		.image_slice_pitch = 0,
		.num_mip_levels = 0,
		.num_samples = 0,
		.buffer= NULL
	};

	cl_mem out_texture = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_format, &image_desc, NULL, &clErr);
	handleClError(clErr, "out clCreateImage");

	return out_texture;
}

int main()
{
	// Host/device data structures
	cl_int clErr;
	cl_context context;
	cl_command_queue queue;

	cl_device_id device = getPreferredDevice();

	// Create a context
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &clErr);
	handleClError(clErr, "clCreateContext");

	cl_program program = buildProgramFromFile(context, device, PROGRAM_NAME".cl", NULL);

	// Create the kernel
	cl_kernel kernel = clCreateKernel(program, PROGRAM_NAME, &clErr);
	handleClError(clErr, "clCreateKernel");

	// Create the command queue
	queue = clCreateCommandQueue(context, device, 0, &clErr);
	handleClError(clErr, "clCreateCommandQueue");

	size_t img_size[3], origin[3] = {0};
	cl_mem in_texture = imageFromFile(context, "input.png", img_size);
	cl_mem out_texture = imageOutputBuffer(context, img_size[0], img_size[1]);

	// Set buffer as argument to the kernel
	clErr = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_texture);
	handleClError(clErr, "[0] clSetKernelArg");
	clErr = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_texture);
	handleClError(clErr, "[1] clSetKernelArg");

	cl_event event;
	clErr = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, img_size, NULL, 0, NULL, &event);
	handleClError(clErr, "clEnqueueNDRangeKernel");
	clErr = clWaitForEvents(1, &event);
	handleClError(clErr, "clWaitForEvents");

	unsigned char* out_data = (unsigned char*)malloc(img_size[0] * img_size[1]*4);
	if(!out_data)
	{
		perror("failed to allocate output buffer");
		exit(1);
	}
	
	// Enqueue the kernel execution command
	clErr = clEnqueueReadImage(queue, out_texture, CL_TRUE, origin, img_size, 0, 0, (void*)out_data, 0, NULL, NULL);
	handleClError(clErr, "clEnqueueReadImage");

	stbi_write_png("output.png", img_size[0], img_size[1], 4, out_data, 4*img_size[0]);
	free(out_data);

	printf("Successfully processed image.\n");

	// Deallocate resources
	clReleaseMemObject(in_texture);
	handleClError(clErr, "in clReleaseMemObject");
	clReleaseMemObject(out_texture);
	handleClError(clErr, "out clReleaseMemObject");
	clReleaseCommandQueue(queue);
	handleClError(clErr, "clReleaseCommandQueue");
	clReleaseKernel(kernel);
	handleClError(clErr, "clReleaseKernel");
	clReleaseProgram(program);
	handleClError(clErr, "clReleaseProgram");
	clReleaseContext(context);
	handleClError(clErr, "clReleaseContext");
//	*/
}
