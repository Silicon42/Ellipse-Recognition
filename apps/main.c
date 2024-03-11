#include <stdio.h>
#include <math.h>
#include <CL/cl.h>
#include "cl_error_handlers.h"
#include "cl_boilerplate.h"
#include "stb_image_write.h"

#define KERNEL_SRC_DIR "kernel/src/"
#define INPUT_FNAME "calendar.jpg"//"input.png"
#define OUTPUT_NAME "output"
#define KERNEL_GLOBAL_BUILD_ARGS "-Werror -g -cl-kernel-arg-info"// -cl-single-precision-constant -cl-fast-relaxed-math"// -cl-no-subgroup-ifp
#define HOUGH_ANGLE_RES (1<<11)

// macro to stringify defined literal values
#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

enum rangeMode {
	//PAD		// pad to multiple of work group dimensions
//	EXACT =	0,	// set range and output to range_param
	REL =	1,	// expand/contract range and output relative to input
	DIAG =	2,	// contraction relative to length of diagonal on [0]*, exact on [1], relative on [2], used for hough_lines
//	HALF =	3,	// half input, range_param ignored
};// * this has the effect of removing the ability to catch lines that clip the corners of the image in return for better buffer utilization

struct queueStage {
	const char* name;		// name of the program to be read in for the stage, must match the filename
	const char* build_args;	// args specific to a kernel
	cl_channel_order output_mode;	// determines the number of channels and mode of the stage's output
	char host_readable;		// boolean indicating if the host will read the output buffer, sets a define for switching certain calculations
	enum rangeMode range_mode;	// what mode to calculate the NDRange in
	int range_param[3];		// effects size of the output buffer, see rangeMode above
	//TODO: break this into separate structures, one for building the stages and one for storage
	size_t out_size[3];		// stage's output buffer size gets stored here
	cl_program program;		// stage's program id gets stored here after being built
	cl_kernel kernel;		// stage's kernel id gets stored here
	cl_mem buffer_texture;	// stage's output buffer id gets stored here
};


int main()
{
	cl_int clErr;

	char hough_args[256] = "";
	struct queueStage \
		scharr = {"scharr", "", CL_RGBA, 0, REL, {0}, {954, 532,1}, NULL, NULL, NULL},\
		canny = {"canny", "", CL_RGBA, 0, REL, {0}, {954, 532,1}, NULL, NULL, NULL},\
		hough = {"hough_lines", hough_args, CL_R, 1, DIAG, {1, HOUGH_ANGLE_RES, 0}, {0}, NULL, NULL, NULL};

	// NULL pointer terminated queueStage* array
	struct queueStage* stages[] = {&scharr, &canny, &hough, NULL};

	// get a device to execute on
	cl_device_id device = getPreferredDevice();

	// Create a context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &clErr);
	handleClError(clErr, "clCreateContext");

	// Create the command queue
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &clErr);
	handleClError(clErr, "clCreateCommandQueue");

	// create input buffer, done early to get image size prior to kernel build phase
	cl_image_format img_format = {
		.image_channel_order = CL_R,
		.image_channel_data_type = CL_UNORM_INT8
	};
	size_t img_size[3];
	cl_mem in_texture = imageFromFile(context, INPUT_FNAME, &img_format, img_size);

	//TODO: need to loop here to calc proper image sizes. move switch-case below to this loop
	// special processing for the Hough build args
	unsigned int hough_half_height = sqrt(img_size[0]*img_size[0] + img_size[1]*img_size[1])/2;
	hough_half_height -= hough.range_param[0];
	hough.out_size[1] = hough_half_height*2;
	hough.out_size[0] =  hough.range_param[1];
	hough.out_size[2] = 1;
	//hough.out_size[3] = {hough_half_height*2, hough.range_param[1], img_size[2] + 2*hough.range_param[2]};
	snprintf(hough_args, sizeof(hough_args)," ");/*\
		"-DHOUGH_HALF_HEIGHT=%u -DANGLE_RES=%zu -DIMG_WIDTH=%zu -DIMG_HEIGHT=%zu",\
		hough_half_height,		hough.out_size[1],	img_size[0],	img_size[1]);*/

	// Build programs from source and create kernels from built programs
	//NOTE: all build args must be set by here
	for(int i = 0; stages[i]; ++i)
	{
		char fname[256];
		char args[1024];
		snprintf(fname, sizeof(fname), KERNEL_SRC_DIR "%s.cl", (*stages[i]).name);
		printf("Building %s.cl\n", (*stages[i]).name);
		snprintf(args, sizeof(args), KERNEL_GLOBAL_BUILD_ARGS " %s", (*stages[i]).build_args);	//TODO: implement output flag define
		(*stages[i]).program = buildProgramFromFile(context, device, fname, args);
		(*stages[i]).kernel = clCreateKernel((*stages[i]).program, (*stages[i]).name, &clErr);
		handleClError(clErr, "clCreateKernel");
	}

	// allocate output buffer
	char* out_data = NULL;
	img_format.image_channel_order = CL_RGBA;
	cl_mem out_texture = imageOutputBuffer(context, &out_data, &img_format, hough.out_size);
	
	// Set buffers as arguments to the kernels
	cl_mem last_buffer = in_texture;
	size_t* last_size = img_size;
	for(int i = 0; stages[i]; ++i)
	{
		printf("Setting args for %s.\n", (*stages[i]).name);
		clErr = clSetKernelArg((*stages[i]).kernel, 0, sizeof(cl_mem), &last_buffer);
		handleClError(clErr, "[0] clSetKernelArg");
		int* range_param = (*stages[i]).range_param;
		size_t* out_size = (*stages[i]).out_size;
		switch ((*stages[i]).range_mode)
		{
		case REL:
			out_size[0] = last_size[0] + range_param[0];
			out_size[1] = last_size[1] + range_param[1];
			out_size[2] = last_size[2] + range_param[2];
			break;
		case DIAG:
			// out_size[0];
			//out_size[1] = range_param[1];
			out_size[2] = last_size[2] + range_param[2];
			break;
		default:
			break;
		}
		last_size = out_size;

		if(!(*stages[i]).host_readable)
			(*stages[i]).buffer_texture = imageIntermediateBuffer(context, out_size, (*stages[i]).output_mode);
		else
			(*stages[i]).buffer_texture = out_texture;	//TODO: make this work for multiple output buffers, currently only supports the last one
		clErr = clSetKernelArg((*stages[i]).kernel, 1, sizeof(cl_mem), &(*stages[i]).buffer_texture);
		handleClError(clErr, "[1] clSetKernelArg");
		last_buffer = (*stages[i]).buffer_texture;
	}

	const size_t origin[3] = {0};

	//------ END OF INITIALIZATION ------//
	//------- START OF MAIN LOOP -------//
	//TODO: this eventually should be a camera feed driven loop

	// enqueue kernels to the command queue
	for(int i = 0; stages[i]; ++i)
	{
		size_t range[3];
		range[0] = (*stages[i]).out_size[0];
		range[1] = (*stages[i]).out_size[1];
		range[2] = (*stages[i]).out_size[2];
		if((*stages[i]).range_mode == DIAG)
			range[1] /= 2;
		printf("Enqueueing %s with range %zu*%zu*%zu.\n", (*stages[i]).name, range[0], range[1], range[2]);
		clErr = clEnqueueNDRangeKernel(queue, (*stages[i]).kernel, 2, NULL, range, NULL, 0, NULL, NULL);
		handleClError(clErr, "clEnqueueNDRangeKernel");
	}

	printf("Processing image.\n");

	// Enqueue a data read back to the host and wait for it to complete
	clErr = clEnqueueReadImage(queue, out_texture, CL_TRUE, origin, hough.out_size, 0, 0, (void*)out_data, 0, NULL, NULL);
	handleClError(clErr, "clEnqueueReadImage");

	// save result, this will eventually be replaced with displaying or other processing
	stbi_write_png(OUTPUT_NAME".png", hough.out_size[0], hough.out_size[1], 4, out_data, 4*hough.out_size[0]);

	//----------- END OF MAIN LOOP -----------//
	//------ START OF DE-INITIALIZATION ------//
	free(out_data);

	printf("Successfully processed image.\n");

	// Deallocate resources
	clReleaseMemObject(in_texture);
	handleClError(clErr, "in clReleaseMemObject");
	for(int i = 0; stages[i]; ++i)
	{
		clReleaseMemObject((*stages[i]).buffer_texture);
		handleClError(clErr, "out clReleaseMemObject");
		clReleaseKernel((*stages[i]).kernel);
		handleClError(clErr, "clReleaseKernel");
		clReleaseProgram((*stages[i]).program);
		handleClError(clErr, "clReleaseProgram");
	}
	clReleaseCommandQueue(queue);
	handleClError(clErr, "clReleaseCommandQueue");
	clReleaseContext(context);
	handleClError(clErr, "clReleaseContext");
}
