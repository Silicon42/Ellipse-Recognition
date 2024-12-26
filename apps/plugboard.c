#include <stdio.h>
#include <math.h>
#include <CL/cl.h>
#include "cl_error_handlers.h"
#include "cl_boilerplate.h"
#include "clbp_error_handling.h"
#include "stb_image_write.h"
#include "clbp_parse_manifest.h"

#define KERNEL_DIR "kernel/"
#define KERNEL_SRC_DIR	KERNEL_DIR"src/"
#define KERNEL_INC_DIR	KERNEL_DIR"inc/"
#define INPUT_FNAME "images/input.png"
#define OUTPUT_NAME "images/output"
// atan2pi() used in gradient direction calc uses infinities internally for horizonal calculations
// Intel CPUs seem to not calculate atan2pi() correctly if -cl-fast-relaxed-math is set and collapse to only either +/- 0.5
#define KERNEL_GLOBAL_BUILD_ARGS "-I"KERNEL_INC_DIR" -Werror -g -cl-kernel-arg-info -cl-single-precision-constant -cl-fast-relaxed-math"
//#define MAX_KERNELS 32
//#define MAX_STAGES 32
//#define MAX_ARGS 200
// macro to stringify defined literal values
#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

//FIXME: need to think of this as a library since we want people to use this to track things
// in their own programs, therefore, it can't be calling exit() in case of an error

int main(int argc, char *argv[])
{
	(void)argc;
	char* in_file = argv[1] ? argv[1] : INPUT_FNAME;
	cl_int clErr;

	// Getting device, context, and command queue done first because if any of these fail, it's likely a higher priority issue
	// than some part of the manifest being invalid since it's likely a hardware or driver issue

	// get a device to execute on
	cl_device_id device = getPreferredDevice();

	// Create a context
	//TODO: this might be better generalized if the device list included all devices for a given platform
//FIXME: specify platform here, right now it just happens to work because it defaults to the first returned platform
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &clErr);
	handleClError(clErr, "clCreateContext");

	// Create the command queue
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &clErr);
	handleClError(clErr, "clCreateCommandQueue");

	// Read + validate MANIFEST.toml to figure out which kernel programs we want compiled,
	// how they should be scheduled, what arguments to feed them, and how those args should be formatted
	// and setup a QStaging object that encapsulates that intent
	clbp_Error e = {.err_code = CLBP_OK};
	toml_table_t* root_tbl = parseManifestFile("MANIFEST.toml", &e);
	handleClBoilerplateError(e);
	QStaging staging;
	allocQStagingArrays(root_tbl, &staging, &e);
	handleClBoilerplateError(e);
	populateQStagingArrays(root_tbl, &staging, &e);
	handleClBoilerplateError(e);

	//TODO: add QStaging caching so that if the manifest isn't changed, we don't have to re-parse everything

	// allocate tracking array for image args
	cl_mem* img_args = malloc(staging.img_arg_cnt * sizeof(cl_mem));
	if(!img_args)
		handleClBoilerplateError((clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "img_args array"});

	// instantiate hard-coded args, this gives us the base image sizes that things are calculated relative to
	cl_image_format img_format = {
		.image_channel_order = CL_R,
		.image_channel_data_type = CL_UNORM_INT8//CL_UNSIGNED_INT8
	};

	img_args[0] = imageFromFile(context, in_file, &img_format, &e);
	if(e.err_code)
		handleClBoilerplateError(e);

	// calculate arg sizes and kernel ranges, this allows baking of image sizes and kernel ranges into kernels if desired
	calcRanges()

	// compile and link kernel programs from source
	cl_program* kprogs = malloc(staging.kernel_cnt * sizeof(cl_program));
	if(!kprogs)
		handleClBoilerplateError((clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "cl_program array"});


/*	cl_kernel* kernels = malloc(staging.kernel_cnt * sizeof(cl_kernel));
	if(!kernels)
		handleClBoilerplateError((clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "cl_kernel array"});*/
	//FIXME: temp fix for OpenCL 1.2 support, add a macro that automatically fixes this
	cl_program linked_prog = buildKernelProgsFromSource(context, device, KERNEL_SRC_DIR, &staging, KERNEL_GLOBAL_BUILD_ARGS, kprogs, &e);
	if(e.err_code)
		handleClBoilerplateError(e);

	// convert the settings into an actual staged queue using the reference kernels generated earlier
	QStage* stages = malloc(staging.stage_cnt * sizeof(QStage));
	if(!stages)
		handleClBoilerplateError((clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "QStage array"});
	
	prepQStages(context, &staging, stages, staging.stage_cnt, &tracker, &e);
	handleClBoilerplateError(e);

	// kernel arguments can't be queried before kernel instantiaion
	//instantiateKernelArgs

	//assignKernelArgs

	//freeStagingArray(staging);
	//free(kernel_progs);
	toml_free(root_tbl);

	// safe to release the context here since it's never used after this point
	clErr = clReleaseContext(context);
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
	for(int i = 0; i < staging.stage_cnt; ++i)
	{
		size_t* range = stages[i].range;
		printf("Enqueueing %s with range %zu*%zu*%zu.\n", staging.kprog_names[i], range[0], range[1], range[2]);
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
	for(int i = 0; i < staging.stage_cnt; ++i)
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
