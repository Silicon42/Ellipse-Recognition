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

	// compile and link kernel programs from source
	// may be moved after calcRanges() if processing size plays a factor in the build process
	// such as if there is only ever a single fixed size that is discovered at runtime
	// tradeoff is it's worse for the memory footprint, but allows for minor optimization for the kernel program
	//TODO: add support for individualized build args
	cl_program linked_prog = buildKernelProgsFromSource(context, device, KERNEL_SRC_DIR, &staging, KERNEL_GLOBAL_BUILD_ARGS, &e);
	handleClBoilerplateError(e);

	//at this point, the arg list and kernel list are finalized and we know how many there will be
	StagedQ staged;
	allocStagedQArrays(&staging, &staged, &e);
	handleClBoilerplateError(e);

	// instantiate hard-coded args, this gives us the base image sizes that things are calculated relative to
	cl_image_format img_format = {
		.image_channel_order = CL_R,
		.image_channel_data_type = CL_UNORM_INT8//CL_UNSIGNED_INT8
	};

	//TODO: see if the instantiation of the mem object can be moved to be at the same time as the others
	staged.img_args[0] = imageFromFile(context, in_file, &img_format, &staged.img_sizes[0], &e);
	handleClBoilerplateError(e);

	// calculate arg sizes and kernel ranges, this allows baking of image sizes and kernel ranges into kernels if desired
	calcRanges(&staging, &staged, &e);
	handleClBoilerplateError(e);

	// kernel arguments can't be queried before kernel instantiaion
	instantiateKernels(context, &staging, linked_prog, &staged, &e);
	handleClBoilerplateError(e);

	// must be run once after first instantiation of kernels and before first instantiation of args
	inferArgAccessAndVerifyFormats(&staging, &staged);

	size_t max_out_sz = instantiateImgArgs(context, &staging, &staged, &e);
	handleClBoilerplateError(e);

	setKernelArgs(context, &staging, &staged, &e);
	handleClBoilerplateError(e);

	// cleanup now that config is fully processed
	freeQStagingArrays(&staging);
	toml_free(root_tbl);
	//TODO: if you add multiple output tracking, then the sizes array of the StagedQ can be freed here

	// safe to release the context here since it's never used after this point
	clErr = clReleaseContext(context);
	handleClError(clErr, "clReleaseContext");

	clErr = clUnloadCompiler();
	handleClError(clErr, "clUnloadCompiler");

	// allocate output buffer
	char* out_data = (char*)malloc(max_out_sz);

	puts("\n");

	//------ END OF INITIALIZATION ------//
	//------- START OF MAIN LOOP -------//
	//TODO: this eventually should be a camera feed driven loop

	// enqueue kernels to the command queue
	for(int i = 0; i < staged.stage_cnt; ++i)
	{
		size_t* range = staged.ranges;//TODO: write a function that returns a version of a Size3D object as size_t array
		printf("Enqueueing %s with range %zu*%zu*%zu.\n", staging.kprog_names[i], range[0], range[1], range[2]);
		clErr = clEnqueueNDRangeKernel(queue, staged.kernels[i], 2, NULL, range, NULL, 0, NULL, NULL);
		handleClError(clErr, "clEnqueueNDRangeKernel");
	}

	printf("\nProcessing image.\n");
	//clFinish(queue);
	uint16_t* out_sz = staged.img_sizes[staged.img_arg_cnt-1].d;
	size_t region[3] = {out_sz[0], out_sz[1], out_sz[2]};
	// Enqueue a data read back to the host and wait for it to complete
	clErr = clEnqueueReadImage(queue, staged.img_args[staged.img_arg_cnt-1], CL_TRUE, (size_t[3]){0}, region, 0, 0, out_data, 0, NULL, NULL);
	handleClError(clErr, "clEnqueueReadImage");

	uint8_t channel_cnt = readImageAsCharArr(out_data, &staged, staged.img_arg_cnt-1);

	// save result
	//TODO: replace this with displaying or other processing
	//NOTE: if channel_cnt == 2, then this gets interpreted as gray + alpha so may look strange simply viewing it
	stbi_write_png(OUTPUT_NAME".png", out_sz[0], out_sz[1], channel_cnt, out_data, channel_cnt*out_sz[0]);

	//----------- END OF MAIN LOOP -----------//
	//------ START OF DE-INITIALIZATION ------//
	printf("\nSuccessfully processed image.\n");

	// Deallocate resources
	free(out_data);
	freeStagedQArrays(&staged);

	clReleaseCommandQueue(queue);
	handleClError(clErr, "clReleaseCommandQueue");
}
