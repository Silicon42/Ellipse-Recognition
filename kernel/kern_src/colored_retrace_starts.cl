// redraws the initial starts after colored_retrace since some may have gotten
// overwritten by the end pixel of other threads, should be scheduled based on 
// dims of start_info input
#include "colorizer.cl_h"
//#include "cast_helpers.cl_h"

kernel void colored_retrace_starts(
	read_only image1d_t is2_start_info,
	write_only image2d_t uc4_trace_image)
{
	short index = get_global_id(0);	// must be scheduled as 1D

	// initialize variables of arcs segment tracing loop for first iteration
	int2 coords = read_imagei(is2_start_info, index).lo;
	uint3 base_color = scatter_colorize(coords.x ^ (coords.x * coords.y));
	//only populated items in the array need to be processed
	if(all(coords == 0))
		return;

	write_imageui(uc4_trace_image, coords, (uint4)(256-base_color,-1) );
}