#include <stdio.h>
#include <math.h>
#include <CL/cl.h>
#include "cl_error_handlers.h"
#include "cl_boilerplate.h"
#include "stb_image_write.h"

#define KERNEL_SRC_DIR "kernel/src/"
#define INPUT_FNAME "images/input.png"
#define OUTPUT_NAME "images/output"
#define KERNEL_GLOBAL_BUILD_ARGS "-Werror -g -cl-kernel-arg-info -cl-single-precision-constant -cl-fast-relaxed-math"// -cl-no-subgroup-ifp
#define HOUGH_ANGLE_RES (1<<11)
#define MAX_KERNELS 8
#define MAX_STAGES 8
#define MAX_ARGS 64
// macro to stringify defined literal values
#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)


int main()
{
	cl_int clErr;
	const char* kernel_progs[] = {"scharr", "canny", "hough_lines", "peaks", NULL};
/*
	const char* kernel_progs[] = {"scharr", "canny", "hough_lines", NULL};
	QStage \
		scharr = {"scharr", CL_RGBA, 0, REL, {0}},\
		canny = {"canny", CL_RGBA, 0, REL, {0}},\
		hough = {"hough_lines", CL_R, 1, DIAG, {HOUGH_ANGLE_RES, -1, 0}};
*/
	// NULL pointer terminated QStage* array
//	QStaging* staging[] = {&scharr, &canny, &hough, NULL};

	// get a device to execute on
	cl_device_id device = getPreferredDevice();

	// Create a context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &clErr);
	handleClError(clErr, "clCreateContext");

	// Create the command queue
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &clErr);
	handleClError(clErr, "clCreateCommandQueue");

	// Safe to release context since the queue now has a reference, this way we don't need to do it later
	//clReleaseContext(context);
	handleClError(clErr, "clReleaseContext");

	//TODO: move this block to a function for initiallizing an ArgTracker since some of these values should always be the same
	// create input buffer, done early to get image size prior to kernel build phase
	TrackedArg ta[MAX_ARGS];
	ArgTracker tracker = {.args = ta, .args_cnt = 1, .max_args = MAX_ARGS, .max_out_size = 0};
	cl_image_format img_format = {
		.image_channel_order = CL_R,
		.image_channel_data_type = CL_UNORM_INT8
	};
	tracker.args[0].format = img_format;
	imageFromFile(context, INPUT_FNAME, &tracker.args[0]);

/*	//TODO: need to loop here to calc proper image sizes. move switch-case below to this loop
	// special processing for the Hough build args
	unsigned int hough_half_height = sqrt(img_size[0]*img_size[0] + img_size[1]*img_size[1])/2;
	hough_half_height -= hough.range_param[0];
	hough.range[1] = hough_half_height*2;
	hough.range[0] =  hough.range_param[1];
	hough.range[2] = 1;
*/
	/*cl_kernel* kernels = (cl_kernel*)malloc(sizeof(cl_kernel)*kernel_cnt);
	if(kernels == NULL)
	{
		perror("Couldn't allocate kernel array");
		exit(1);
	}*/

	// build reference kernels from source
	cl_kernel kernels[MAX_KERNELS];
	cl_uint kernel_cnt = buildKernelsFromSource(context, device, KERNEL_SRC_DIR, kernel_progs, KERNEL_GLOBAL_BUILD_ARGS, kernels, MAX_KERNELS);

	// staged queue settings of which kernels to use and how
	ArgStaging simple[2] = {{1,{REL,{0}},CL_FALSE,CL_FALSE},{1,{REL,{0}},CL_FALSE,CL_FALSE}};
	ArgStaging diag[2] = {{1,{REL,{0}},CL_FALSE,CL_FALSE},{1,{DIAG,{2048, -4, 0}},CL_FALSE,CL_FALSE}};
	ArgStaging out[2] = {{1,{REL,{0}},CL_FALSE,CL_FALSE},{1,{REL,{0}},CL_TRUE,CL_FALSE}};
	const QStaging* staging[] = {
		&(QStaging){0, 2, {REL, {0}}, simple},
		&(QStaging){1, 2, {REL, {0}}, simple},
		&(QStaging){2, 1, {DIVIDE, {1, 2, 1}}, diag},
		&(QStaging){3, 2, {REL, {0, -2, 0}}, out},
		NULL
	};

	// convert the settings into an actual staged queue using the reference kernels generated earlier
	QStage stages[MAX_STAGES];
	int stage_cnt = prepQStages(context, staging, kernels, stages, MAX_STAGES, &tracker);

	// release the reference kernels when done with staging
	for(cl_uint i = 0; i < kernel_cnt; ++i)
	{
		clReleaseKernel(kernels[i]);
		handleClError(clErr, "clReleaseKernel");
	}
	// allocate output buffer
	void* out_data = malloc(tracker.max_out_size);
	//img_format.image_channel_order = CL_RGBA;
	//cl_mem out_texture = imageOutputBuffer(context, &out_data, &img_format, hough.range);

/*	// Set buffers as arguments to the kernels
	cl_mem last_buffer = in_texture;
	size_t* last_size = img_size;
	for(int i = 0; stages[i]; ++i)
	{
		printf("Setting args for %s.\n", (*stages[i]).name);
		clErr = clSetKernelArg((*stages[i]).kernel, 0, sizeof(cl_mem), &last_buffer);
		handleClError(clErr, "[0] clSetKernelArg");
		int* range_param = (*stages[i]).range_param;
		size_t* range = (*stages[i]).range;
		switch ((*stages[i]).range_mode)
		{
		case REL:
			range[0] = last_size[0] + range_param[0];
			range[1] = last_size[1] + range_param[1];
			range[2] = last_size[2] + range_param[2];
			break;
		case DIAG:
			// range[0];
			//range[1] = range_param[1];
			range[2] = last_size[2] + range_param[2];
			break;
		default:
			break;
		}
		last_size = range;

		if(!(*stages[i]).host_readable)
			(*stages[i]).buffer_texture = imageIntermediateBuffer(context, range, (*stages[i]).output_mode);
		else
			(*stages[i]).buffer_texture = out_texture;	//TODO: make this work for multiple output buffers, currently only supports the last one
		clErr = clSetKernelArg((*stages[i]).kernel, 1, sizeof(cl_mem), &(*stages[i]).buffer_texture);
		handleClError(clErr, "[1] clSetKernelArg");
		last_buffer = (*stages[i]).buffer_texture;
	}
	*/
	const size_t origin[3] = {0};

	//------ END OF INITIALIZATION ------//
	//------- START OF MAIN LOOP -------//
	//TODO: this eventually should be a camera feed driven loop

	// enqueue kernels to the command queue
	for(int i = 0; i < stage_cnt; ++i)
	{
		size_t* range = stages[i].range;
//		if(stages[i].range_mode == DIAG)
//			range[1] /= 2;
		printf("Enqueueing %s with range %zu*%zu*%zu.\n", stages[i].name, range[0], range[1], range[2]);
		clErr = clEnqueueNDRangeKernel(queue, stages[i].kernel, 2, NULL, range, NULL, 0, NULL, NULL);
		handleClError(clErr, "clEnqueueNDRangeKernel");
	}

	printf("Processing image.\n");

	size_t* last_size = tracker.args[tracker.args_cnt - 1].size;
	// Enqueue a data read back to the host and wait for it to complete
	clErr = clEnqueueReadImage(queue, tracker.args[tracker.args_cnt - 1].arg, CL_TRUE, origin, last_size, 0, 0, out_data, 0, NULL, NULL);
	handleClError(clErr, "clEnqueueReadImage");

	// save result
	//TODO: replace this with displaying or other processing
	stbi_write_png(OUTPUT_NAME".png", last_size[0], last_size[1], 1, out_data, 1*last_size[0]);

	//----------- END OF MAIN LOOP -----------//
	//------ START OF DE-INITIALIZATION ------//
	free(out_data);

	printf("Successfully processed image.\n");

	// Deallocate resources
	//clReleaseMemObject(in_texture);
	//handleClError(clErr, "in clReleaseMemObject");
	for(int i = 0; i < stage_cnt; ++i)
	{
		clReleaseKernel(stages[i].kernel);
		handleClError(clErr, "clReleaseKernel");
	}

	for(int i = 0; i < tracker.args_cnt; ++i)
	{
		clReleaseMemObject(tracker.args[i].arg);
		handleClError(clErr, "clReleaseMemObject");
	}

	clReleaseCommandQueue(queue);
	handleClError(clErr, "clReleaseCommandQueue");
}
