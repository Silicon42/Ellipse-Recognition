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
#define ALLOCATION_ERROR "\nERROR: Failed to allocate %s.\n"
#define MANIFEST_ERROR "\nMANIFEST ERROR: "
// atan2pi() used in gradient direction calc uses infinities internally for horizonal calculations
// Intel CPUs seem to not calculate atan2pi() correctly if -cl-fast-relaxed-math is set and collapse to only either +/- 0.5
#define KERNEL_GLOBAL_BUILD_ARGS "-I"KERNEL_INC_DIR" -Werror -g -cl-kernel-arg-info -cl-single-precision-constant -cl-fast-relaxed-math"
//#define MAX_KERNELS 32
//#define MAX_STAGES 32
//#define MAX_ARGS 200
// macro to stringify defined literal values
#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

// calloc() wrapper that also handles error reporting and calls exit(1) in case of failure
void* critical_calloc(size_t numOfElements, size_t sizeOfElements, const char* name)
{
	void* ptr = calloc(numOfElements, sizeOfElements);
	if(ptr)
		return ptr;
	//else
	fprintf(stderr, ALLOCATION_ERROR, name);
	exit(1);
}

// malloc() wrapper that also handles error reporting and calls exit(1) in case of failure
void* critical_malloc(size_t numBytes, const char* name)
{
	void* ptr = malloc(numBytes);
	if(ptr)
		return ptr;
	//else
	fprintf(stderr, ALLOCATION_ERROR, name);
	exit(1);
}

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
	char errbuf[256];
	toml_table_t* root_tbl = toml_parse(manifest, errbuf, sizeof(errbuf));
	free(manifest);	// parsing works for whole document, so c-string is no longer needed
	if (!root_tbl)
	{
		fprintf(stderr, "\nTOML ERROR: %s\n", errbuf);
		exit(1);
	}

	// get stages array and check valid size and type
	toml_array_t* stage_list = toml_table_array(root_tbl, "stages");
	if(!stage_list || !stage_list->nitem)
	{
		perror(MANIFEST_ERROR"no stages specified.\n");
		exit(1);
	}
	if(stage_list->kind != 't')
	{
		perror(MANIFEST_ERROR"stages array must be a table array.\n");
		exit(1);
	}
	int stage_cnt = stage_list->nitem;

	// get arg table
	toml_table_t* args_table = toml_table_table(root_tbl, "args");
	if(!args_table || !args_table->nkval)
	{
		perror(MANIFEST_ERROR"args table was empty.\n");
		exit(1);
	}
	int max_defined_args = args_table->nkval;

	int kprog_cnt = 0;
	const char** kernel_progs = critical_calloc(stage_cnt + 1, sizeof(char*), "kernel program name array");	//calloc ensures unset values are null
	QStaging* staging = critical_malloc(stage_cnt * sizeof(QStaging), "kernel program staging array");
	const char** arg_names = critical_calloc(max_defined_args + 1, sizeof(char*), "arg name array");
	ArgStaging* arg_stg = critical_malloc(max_defined_args * sizeof(ArgStaging), "arg staging array");
	int last_arg_idx = 0;

	for(int i = 0; i < stage_cnt; ++i)
	{
		toml_table_t* stage = toml_array_table(stage_list, i);	//can't return null since we already have valid stage count
		toml_value_t tval = toml_table_string(stage, "name");
		if(!tval.ok || !tval.u.s[0])	//not sure if this is safe or if the compiler might do them in an unsafe order
		{
			fprintf(stderr, MANIFEST_ERROR"missing name field at entry %i of stages array.\n", i);
			exit(1);
		}

		// check if a kernel by that name already exists, if not, add it to the list of ones to build
		staging[i].kernel_idx = addUniqueString(kernel_progs, stage_cnt, tval.u.s);

		toml_array_t* args = toml_table_array(stage, "args");
		if(!args || !args->nitem)
		{
			fprintf(stderr, MANIFEST_ERROR"missing args list at entry %i of stages array.\n", i);
			exit(1);
		}
		if(args->kind != 'v' || args->type != 's')	// might be sufficient to only check type
		{
			fprintf(stderr, MANIFEST_ERROR"args array at entry %i of stages array must contain only strings.\n", i);
			exit(1);
		}
		int args_cnt = args->nitem;

		staging[i].arg_idxs = critical_malloc(args_cnt * sizeof(int), "stage's arg index list");

		// iterate over args to find any new ones
		for(int j = 0; j < args_cnt; ++j)
		{
			char* arg_name = toml_array_string(args, j).u.s;	//guaranteed exists due kind, type, and count checks above
			if(arg_name[0])	// if not empty string
			{
				int arg_idx = addUniqueString(arg_names, max_defined_args, arg_name);
				staging[i].arg_idxs[j] = arg_idx;
				if(arg_idx > last_arg_idx)	//check if this was a newly referenced argument
				{
					++last_arg_idx;	//is only ever bigger by one so this is safe
					//instantiate a corresponding arg on the arg staging array
					toml_table_t* arg_conf = toml_table_table(args, arg_name);
					if(!arg_conf)
					{
						fprintf(stderr, MANIFEST_ERROR"stage %i requested \"%s\" but no such key was found under [args].\n", i, arg_name);
						exit(1);
					}
					

				}
			}
			else	// empty string is a special case that always selects whatever was last added
				staging[i].arg_idxs[j] = last_arg_idx;
		}
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

	freeStagingArray(staging);
	free(kernel_progs);
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
