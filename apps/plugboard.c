#include <stdio.h>
#include <math.h>
#include <CL/cl.h>
#include "cl_error_handlers.h"
#include "cl_boilerplate.h"
#include "stb_image_write.h"
#include "toml-c.h"

#define KERNEL_DIR "kernel/"
#define KERNEL_SRC_DIR	KERNEL_DIR"src/"
#define KERNEL_INC_DIR	KERNEL_DIR"inc/"
#define INPUT_FNAME "images/input.png"
#define OUTPUT_NAME "images/output"
// atan2pi() used in gradient direction calc uses infinities internally for horizonal calculations
// Intel CPUs seem to not calculate atan2pi() correctly if -cl-fast-relaxed-math is set and collapse to only either +/- 0.5
#define KERNEL_GLOBAL_BUILD_ARGS "-I"KERNEL_INC_DIR" -Werror -g -cl-kernel-arg-info -cl-single-precision-constant -cl-fast-relaxed-math"
#define MAX_KERNELS 32
#define MAX_STAGES 32
#define MAX_ARGS 64
// macro to stringify defined literal values
#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)


int main(int argc, char *argv[])
{
	(void)argc;
	char* in_file = argv[1] ? argv[1] : INPUT_FNAME;

	cl_int clErr;

	// get a device to execute on
	cl_device_id device = getPreferredDevice();

	// Create a context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &clErr);
	handleClError(clErr, "clCreateContext");

	// Create the command queue
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &clErr);
	handleClError(clErr, "clCreateCommandQueue");

	// Read in the manifest for what kernels should be used
	char* manifest = readFileToCstring(KERNEL_DIR"MANIFEST.toml");
	char errbuf[200];
	toml_table_t* root_tbl = toml_parse(manifest, errbuf, sizeof(errbuf));
	free(manifest);	// parsing works for whole document, so c-string is no longer needed
	if (!root_tbl)
	{
		fprintf(stderr, "\nERROR: %s\n", errbuf);
		exit(1);
	}

	// read contents of stages array
	toml_array_t* arr = toml_table_array(root_tbl, "stages");
	int arr_len = toml_array_len(arr);
	int kprog_cnt = 0;
	const char** kernel_progs = calloc((arr_len + 1), sizeof(char*));
	if(!kernel_progs)
	{
		perror("\nERROR: Couldn't allocate kernel program list.\n");
		exit(1);
	}
	const char* arg_list[MAX_ARGS];

	for(int i = 0; i < arr_len; ++i)
	{
		toml_table_t* stage = toml_array_table(arr, i);
		toml_value_t tval = toml_table_string(stage, "name");
		if(!tval.ok)
		{
			fprintf(stderr, "\nMANIFEST ERROR: name required at entry %i of stages array\n", i);
			exit(1);
		}

		// check if a kernel by that name already exists, if not, add it to the list of ones to build
		addUniqueString(kernel_progs, arr_len, tval.u.s);

		//TODO: add support for separate config files, currently only supports inline and default
		
	}

	//TODO: move this block to a function for initiallizing an ArgTracker since some of these values should always be the same
	// create input buffer, done early to get image size prior to kernel build phase
	TrackedArg ta[MAX_ARGS];
	ArgTracker tracker = {.args = ta, .args_cnt = 1, .max_args = MAX_ARGS, .max_out_size = 0};
	cl_image_format img_format = {
		.image_channel_order = CL_R,
		.image_channel_data_type = CL_UNORM_INT8//CL_UNSIGNED_INT8
	};
	tracker.args[0].format = img_format;
	imageFromFile(context, in_file, &tracker.args[0]);

	// build reference kernels from source
	cl_kernel kernels[MAX_KERNELS];
	//FIXME: temp fix for OpenCL 1.2 support
	/*	cl_uint kernel_cnt = */buildKernelsFromSource(context, device, KERNEL_SRC_DIR, kernel_progs, KERNEL_GLOBAL_BUILD_ARGS, kernels, MAX_KERNELS);

	// convert the settings into an actual staged queue using the reference kernels generated earlier
	QStage stages[MAX_STAGES];
	int stage_cnt = prepQStages(context, staging, kernels, stages, MAX_STAGES, &tracker);

	toml_free(root_tbl);

	// safe to release the context here since it's never used after this point
	clReleaseContext(context);
	handleClError(clErr, "clReleaseContext");

	// allocate output buffer
	char* out_data = (char*)malloc(tracker.max_out_size);

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
	//clFinish(queue);
	size_t* last_size = tracker.args[tracker.args_cnt - 1].size;
	// Enqueue a data read back to the host and wait for it to complete
	clErr = clEnqueueReadImage(queue, tracker.args[tracker.args_cnt - 1].arg, CL_TRUE, origin, last_size, 0, 0, out_data, 0, NULL, NULL);
	handleClError(clErr, "clEnqueueReadImage");

	unsigned char channel_cnt = readImageAsCharArr(out_data, &tracker.args[tracker.args_cnt - 1]);

	// save result
	//TODO: replace this with displaying or other processing
	//NOTE: if channel_cnt == 2, then this gets interpreted as gray + alpha so may look strange simply viewing it
	stbi_write_png(OUTPUT_NAME".png", last_size[0], last_size[1], channel_cnt, out_data, channel_cnt*last_size[0]);

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
