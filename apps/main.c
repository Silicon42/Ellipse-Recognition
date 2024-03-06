#include <stdio.h>
#include <CL/cl.h>
#include "cl_error_handlers.h"
#include "cl_boilerplate.h"
#include "stb_image_write.h"

#define KERNEL_SRC_DIR "kernel/src/"
#define STAGE_CNT 3
#define INPUT_FNAME "input.png"
#define OUTPUT_NAME "output"

// macro to stringify defined literal values
#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)


struct queueStage {
	const char* name;		// name of the program to be read in for the stage, must match the filename
	cl_channel_order channel_mode;	// determines the number of channels and mode of the stage's output
	cl_program program;		// where the stage's program id gets kept after being built
	cl_kernel kernel;		// where the stage's kernel id is kept
	cl_mem buffer_texture;	// where the stage's output buffer id is kept
};


int main()
{
	cl_int clErr;
	struct queueStage stages[STAGE_CNT] = {
		{"scharr",	CL_RG,	NULL, NULL, NULL},
		{"mag_ang",	CL_RGBA,NULL, NULL, NULL},
		{"canny",	CL_R,	NULL, NULL, NULL}
	};

	// get a device to execute on
	cl_device_id device = getPreferredDevice();

	// Create a context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &clErr);
	handleClError(clErr, "clCreateContext");

	// Create the command queue
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &clErr);
	handleClError(clErr, "clCreateCommandQueue");

	// Build programs from source and create kernels from built programs
	for(int i = 0; i < STAGE_CNT; ++i)
	{
		char fname[260];
		snprintf(fname, sizeof(fname), KERNEL_SRC_DIR "%s.cl", stages[i].name);
		printf("Building %s.cl\n", stages[i].name);
		stages[i].program = buildProgramFromFile(context, device, fname, NULL);
		stages[i].kernel = clCreateKernel(stages[i].program, stages[i].name, &clErr);
		handleClError(clErr, "clCreateKernel");
	}

	// create buffers
	size_t img_size[3], origin[3] = {0};
	cl_mem in_texture = imageFromFile(context, INPUT_FNAME, img_size);
	cl_mem out_texture = imageOutputBuffer(context, img_size);

	// allocate output buffer
	unsigned char* out_data = (unsigned char*)malloc(img_size[0] * img_size[1]*4);
	if(!out_data)
	{
		perror("failed to allocate output buffer");
		exit(1);
	}
	
	// Set buffer as arguments to the kernels
	cl_mem last_buffer = in_texture;
	for(int i = 0; i < STAGE_CNT; ++i)
	{
		clErr = clSetKernelArg(stages[i].kernel, 0, sizeof(cl_mem), &last_buffer);
		handleClError(clErr, "[0] clSetKernelArg");
		if(i < STAGE_CNT - 1)	// if not the last stage
			stages[i].buffer_texture = imageIntermediateBuffer(context, img_size, stages[i].channel_mode);
		else
			stages[i].buffer_texture = out_texture;
		clErr = clSetKernelArg(stages[i].kernel, 1, sizeof(cl_mem), &stages[i].buffer_texture);
		handleClError(clErr, "[1] clSetKernelArg");
		last_buffer = stages[i].buffer_texture;
	}

	//------ END OF INITIALIZATION ------//
	//------ START OF MAIN LOOP ------//
	//TODO: this eventually should be a camera feed driven loop

	// enqueue kernels to the command queue
	for(int i = 0; i < STAGE_CNT; ++i)
	{
		clErr = clEnqueueNDRangeKernel(queue, stages[i].kernel, 2, NULL, img_size, NULL, 0, NULL, NULL);
		handleClError(clErr, "clEnqueueNDRangeKernel");
	}

	// Enqueue a data read back to the host and wait for it to complete
	clErr = clEnqueueReadImage(queue, out_texture, CL_TRUE, origin, img_size, 0, 0, (void*)out_data, 0, NULL, NULL);
	handleClError(clErr, "clEnqueueReadImage");

	// save result, this will eventually be replaced with displaying or other processing
	stbi_write_png(OUTPUT_NAME".png", img_size[0], img_size[1], 4, out_data, 4*img_size[0]);

	//------ END OF MAIN LOOP ------//
	//------ START OF DE-INITIALIZATION ------//
	free(out_data);

	printf("Successfully processed image.\n");

	// Deallocate resources
	clReleaseMemObject(in_texture);
	handleClError(clErr, "in clReleaseMemObject");
	for(int i = 0; i < STAGE_CNT; ++i)
	{
		clReleaseMemObject(stages[i].buffer_texture);
		handleClError(clErr, "out clReleaseMemObject");
		clReleaseKernel(stages[i].kernel);
		handleClError(clErr, "clReleaseKernel");
		clReleaseProgram(stages[i].program);
		handleClError(clErr, "clReleaseProgram");
	}
	clReleaseCommandQueue(queue);
	handleClError(clErr, "clReleaseCommandQueue");
	clReleaseContext(context);
	handleClError(clErr, "clReleaseContext");
}
