#include <stdio.h>
#include <CL/cl.h>
#include "common_error_handlers.h"
#include "cl_boilerplate.h"
#include "stb_image_write.h"

#define KERNEL_SRC_DIR "kernel/src/"
#define PROGRAM_NAME "scharr"
#define INPUT_FNAME "input.png"
#define OUTPUT_NAME "output"

// macro to stringify defined literal values
//#define STR_EXPAND(tok) #tok
//#define STR(tok) STR_EXPAND(tok)


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

	cl_program program = buildProgramFromFile(context, device, KERNEL_SRC_DIR PROGRAM_NAME".cl", NULL);

	// Create the kernel
	cl_kernel kernel = clCreateKernel(program, PROGRAM_NAME, &clErr);
	handleClError(clErr, "clCreateKernel");

	// Create the command queue
	queue = clCreateCommandQueue(context, device, 0, &clErr);
	handleClError(clErr, "clCreateCommandQueue");

	size_t img_size[3], origin[3] = {0};
	cl_mem in_texture = imageFromFile(context, INPUT_FNAME, img_size);
	cl_mem out_texture = imageOutputBuffer(context, img_size);

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

	stbi_write_png(OUTPUT_NAME".png", img_size[0], img_size[1], 4, out_data, 4*img_size[0]);
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
