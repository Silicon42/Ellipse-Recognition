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
	const char* kernel_progs[] = {"scharr", "canny", "hough_lines", "peaks", "inv_hough_lines", NULL};

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

	// build reference kernels from source
	cl_kernel kernels[MAX_KERNELS];
	cl_uint kernel_cnt = buildKernelsFromSource(context, device, KERNEL_SRC_DIR, kernel_progs, KERNEL_GLOBAL_BUILD_ARGS, kernels, MAX_KERNELS);

	// staged queue settings of which kernels to use and how
	ArgStaging simple[2] = {{1,{REL,{0}},CL_FALSE,CL_FALSE},{1,{REL,{0}},CL_FALSE,CL_FALSE}};
	ArgStaging diag[2] = {{1,{REL,{0}},CL_FALSE,CL_FALSE},{1,{DIAG,{2048, -4, 0}},CL_FALSE,CL_FALSE}};
	ArgStaging out[2] = {{1,{REL,{0}},CL_FALSE,CL_FALSE},{3,{REL,{0}},CL_TRUE,CL_FALSE}};
	const QStaging* staging[] = {
		&(QStaging){0, 2, {REL, {0}}, simple},
		&(QStaging){1, 2, {REL, {0}}, simple},
		&(QStaging){2, 1, {DIVIDE, {1, 2, 1}}, diag},
		&(QStaging){3, 2, {REL, {0, -2, 0}}, simple},
		&(QStaging){4, 2, {REL, {0}}, out},
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

	clErr = clUnloadCompiler();
	handleClError(clErr, "clUnloadCompiler");

	puts("\n");
	const size_t origin[3] = {0};

	//------ END OF INITIALIZATION ------//
	//------- START OF MAIN LOOP -------//
	//TODO: this eventually should be a camera feed driven loop

	// enqueue kernels to the command queue
	for(int i = 0; i < stage_cnt; ++i)
	{
		size_t* range = stages[i].range;
		printf("Enqueueing %s with range %zu*%zu*%zu.\n", stages[i].name, range[0], range[1], range[2]);
		clErr = clEnqueueNDRangeKernel(queue, stages[i].kernel, 2, NULL, range, NULL, 0, NULL, NULL);
		handleClError(clErr, "clEnqueueNDRangeKernel");
	}

	printf("\nProcessing image.\n");

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

	printf("\nSuccessfully processed image.\n");

	// Deallocate resources
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
