#include <stdio.h>
#include <math.h>
#include <CL/cl.h>
#include "cl_error_handlers.h"
#include "cl_boilerplate.h"
#include "stb_image_write.h"

#define KERNEL_SRC_DIR "kernel/src/"
#define INPUT_FNAME "images/input.png"
#define OUTPUT_NAME "images/output"
// atan2pi() used in gradient direction calc uses infinities internally for horizonal calculations
// Intel CPUs seem to not calculate atan2pi() correctly if -cl-fast-relaxed-math is set and collapse to only either +/- 0.5
#define KERNEL_GLOBAL_BUILD_ARGS "-Ikernel/inc -Werror -g -cl-kernel-arg-info -cl-single-precision-constant -cl-fast-relaxed-math"
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
	const char* kernel_progs[] = {
//		"robertsX_char",
		"scharr3_char",
		"non_max_sup",
		"edge_thinning",
		"edge_thinning2",
		"link_edge_pixels",
		"find_segment_starts",
		"starts_debug",
		"serial_reduce",
		"line_segments",
		"segment_debug",
		"gradient_debug",
		"colored_retrace",
		"lost_seg_debug",
		"link_debug",
		"starts_link_debug",
		"colored_retrace_starts",
		"serial_reduce_lines",
		//"arc_adj_matrix",
		"arc_adj_debug",	//temporary dummy to prevent index #s from changing
		"arc_adj_debug",
		"search_region_test",
		"arc_builder",
		"foci_debug",
		NULL
	};

	// get a device to execute on
	cl_device_id device = getPreferredDevice();

	// Create a context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &clErr);
	handleClError(clErr, "clCreateContext");

	// Create the command queue
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &clErr);
	handleClError(clErr, "clCreateCommandQueue");

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

	// staged queue settings of which kernels to use and how
	////// main settings //////
/*	ArgStaging simple_shrink1[] = {
		{1,{REL,{0}},CL_FALSE,CL_FALSE},
		{1,{REL,{-1,-1,0}},CL_TRUE,CL_FALSE}
	};
*/	ArgStaging simple[] = {
		{1,{REL,{0}},CL_FALSE,CL_FALSE},
		{1,{REL,{0}},CL_TRUE, CL_FALSE}
	};
	ArgStaging starts[] = {
		{1,{REL,{0}},CL_FALSE,CL_FALSE},	//uc1_cont
		{2,{REL,{0}},CL_FALSE,CL_FALSE},	//iC1_grad_ang
		{1,{REL,{0}},CL_TRUE, CL_FALSE}		//uc1_starts_cont
	};
	ArgStaging serial[] = {
		{1,{REL,{0}},CL_FALSE,CL_FALSE},			//uc1_starts_cont
		{1,{EXACT,{16384,1,1}},CL_TRUE,CL_FALSE}	//iS2_start_coords
	};
	ArgStaging line_segments[] = {
		{1,{REL,{0}},CL_FALSE,CL_FALSE},	//iS2_start_coords
		{2,{REL,{0}},CL_FALSE,CL_FALSE},	//uc1_cont_info
		{1,{REL,{0}},CL_TRUE, CL_FALSE},	//us1_line_counts
		{4,{REL,{0}},CL_TRUE, CL_FALSE}		//iC2_line_data
	};
	ArgStaging serial2[] = {	//serial_reduce_lines
		{3,{REL,{0}},CL_FALSE,CL_FALSE},			//iS2_start_coords
		{1,{REL,{0}},CL_FALSE,CL_FALSE},			//ui4_line_data
		{2,{REL,{0}},CL_FALSE,CL_FALSE},			//us1_line_counts
		{1,{EXACT,{1,1,1}},CL_TRUE,CL_FALSE},		//us1_length
		{1,{EXACT,{256,256,1}},CL_TRUE,CL_FALSE}	//iS2_line_coords
	};
	ArgStaging arc_adj[] = {
		{3,{REL,{0}},CL_FALSE,CL_FALSE},	//iC2_line_data
		{1,{REL,{0}},CL_FALSE,CL_FALSE},	//iS2_line_coords
		{2,{REL,{0}},CL_FALSE,CL_FALSE},	//us1_length
		{1,{REL,{0}},CL_TRUE, CL_FALSE}		//us4_sparse_adj_matrix
	};
	ArgStaging arc_builder[] = {	//arc_builder
		{3,{REL,{0}},CL_FALSE,CL_FALSE},			//iS2_start_coords
		{2,{REL,{0}},CL_FALSE,CL_FALSE},			//us1_line_counts
		{1,{REL,{0}},CL_FALSE,CL_FALSE},			//ui4_line_data
		{1,{REL,{0}},CL_TRUE,CL_FALSE},				//us1_line_counts
		{1,{REL,{0}},CL_TRUE,CL_FALSE}				//iS2_line_coords
	};

	////// debug settings //////
	ArgStaging mul3[] = {
		{1,{REL,{0}},CL_FALSE,CL_FALSE},
		{1,{MULT,{3,3,1}},CL_TRUE,CL_FALSE}
	};
	ArgStaging starts_debug[] = {
		{3,{REL,{0}},CL_FALSE,CL_FALSE},	//iC1_thin
		{1,{REL,{0}},CL_FALSE,CL_FALSE},	//uc1_seg_start
		{1,{REL,{0}},CL_TRUE, CL_FALSE}		//uc4_dst_image
	};
/*	ArgStaging segment_debug[] = {	//DEPRECATED
		{7,{REL,{0}},CL_FALSE,CL_FALSE},
		{6,{REL,{0}},CL_FALSE,CL_FALSE},
		{5,{REL,{0}},CL_FALSE,CL_FALSE},
		{3,{REL,{0}},CL_FALSE,CL_FALSE},
		{1,{REL,{0}},CL_FALSE,CL_FALSE},
		{1,{REL,{0}},CL_TRUE, CL_FALSE}
	};
*/	ArgStaging retrace[] = {
		{4,{REL,{0}},CL_FALSE,CL_FALSE},	//uc1_cont_info
		{3,{REL,{0}},CL_FALSE,CL_FALSE},	//iS2_start_info
		{2,{REL,{0}},CL_FALSE,CL_FALSE},	//us1_line_counts
		{1,{REL,{0}},CL_FALSE,CL_FALSE},	//ui4_line_data
		{1,{REL,{0}},CL_TRUE, CL_FALSE}		//uc4_trace_image
	};
	ArgStaging retrace_starts[] = {
		{4,{REL,{0}},CL_FALSE,CL_FALSE},	//iS2_start_info
		//{3,{REL,{0}},CL_FALSE,CL_FALSE},	//ui4_path_image
		{1,{REL,{0}},CL_TRUE, CL_FALSE}		//uc4_trace_image
	};
	ArgStaging lost_seg[] = {
		{2,{REL,{0}},CL_FALSE,CL_FALSE},	//uc4_retrace
		{1,{REL,{0}},CL_FALSE,CL_FALSE},	//uc4_retrace_starts
		{6,{REL,{0}},CL_FALSE,CL_FALSE},	//uc1_cont_data
		{1,{REL,{0}},CL_TRUE, CL_FALSE}		//uc4_out
	};
	ArgStaging search_debug[] = {
		{2,{REL,{0}},CL_FALSE,CL_FALSE},	//iC2_line_data
		{1,{REL,{0}},CL_FALSE,CL_FALSE},	//uc4_trace_image
		{1,{REL,{0}},CL_TRUE, CL_FALSE}		//uc4_debug
	};
	ArgStaging arc_debug[] = {
		{4,{REL,{0}},CL_FALSE,CL_FALSE},	//iC2_line_data
		{2,{REL,{0}},CL_FALSE,CL_FALSE},	//iS2_line_coords
		{3,{REL,{0}},CL_FALSE,CL_FALSE},	//us1_length
		{1,{REL,{0}},CL_FALSE,CL_FALSE},	//us4_sparse_adj_matrix
		{4,{REL,{0}},CL_TRUE,CL_FALSE}		//uc4_out_image
	};
	ArgStaging foci_debug[] = {
		{3,{REL,{0}},CL_FALSE,CL_FALSE},	//iC2_line_data
		{2,{REL,{0}},CL_FALSE,CL_FALSE},	//us1_seg_in_arc
		{1,{REL,{0}},CL_FALSE,CL_FALSE},	//fF4_ellipse_foci
		{1,{REL,{0}},CL_TRUE,CL_FALSE}		//uc4_out_image
	};

	const QStaging* staging[] = {
//		&(QStaging){0, 1, {REL, {0}}, simple_shrink1},	//RobertsX
		&(QStaging){0, 1, {REL, {0}}, simple},			//Scharr 3*3
		&(QStaging){1, 1, {REL, {0}}, simple},			//Non-Max Suppression
		&(QStaging){2, 1, {REL, {0}}, simple},			//Edge Thinning
		&(QStaging){3, 1, {REL, {0}}, simple},			//Edge Thinning
//		&(QStaging){10, 2, {REL, {0}}, mul3},			//Gradient Debug
		&(QStaging){4, 1, {REL, {0}}, simple},			//Link Edge Pixels
//		&(QStaging){13, 2, {REL, {0}}, mul3},			//Link Debug
		&(QStaging){5, 1, {REL, {0}}, starts},			//Find Segment Starts
//		&(QStaging){14, 2, {REL, {0}}, mul3},			//Starts Link Debug
//		&(QStaging){6, 1, {REL, {0}}, starts_debug},	//Starts Debug
		&(QStaging){7, 1, {EXACT, {1,1,1}}, serial},	//Serial Reduce Starts
		&(QStaging){8, 2, {REL, {0}}, line_segments},	//Line Segments
//		&(QStaging){9, 1, {REL, {0}}, segment_debug},	//Segment Debug				//DEPRECATED
//		&(QStaging){11, 4, {REL, {0}}, retrace},		//Colored Retrace
//		&(QStaging){15, 5, {REL, {0}}, retrace_starts},	//Colored Retrace Starts	//currently non-functioning
//		&(QStaging){12, 1, {REL, {0}}, lost_seg},		//Lost Segment Debug		//currently non-functioning
//		&(QStaging){19, 10, {REL, {0}}, search_debug},	//Search Region Test		//currently non-functioning
/*		&(QStaging){16, 1, {EXACT, {1,1,1}}, serial2},	//Serial Reduce Lines
		&(QStaging){17, 1, {REL, {0}}, arc_adj},		//Arc Adjacency Matrix
		&(QStaging){18, 1, {REL, {0}}, arc_debug},		//Arc Adjacency Debug
*/		&(QStaging){20, 4, {REL, {0}}, arc_builder},	//Arc Builder
		&(QStaging){21, 1, {REL, {0}}, foci_debug},		//Foci Debug
/**/		NULL										////-END-////
	};

	// convert the settings into an actual staged queue using the reference kernels generated earlier
	QStage stages[MAX_STAGES];
	int stage_cnt = prepQStages(context, staging, kernels, stages, MAX_STAGES, &tracker);

	// safe to release the context here since it's never used after this point
	clReleaseContext(context);
	handleClError(clErr, "clReleaseContext");

	// release the reference kernels when done with staging
	/*	//FIXME: temp fix for OpenCL 1.2 support
	for(cl_uint i = 0; i < kernel_cnt; ++i)
	{
		clReleaseKernel(kernels[i]);
		handleClError(clErr, "clReleaseKernel");
	}
	*/
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
