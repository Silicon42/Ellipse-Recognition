// redraws the initial starts after colored_retrace since some may have gotten
// overwritten by the end pixel of other threads, should be scheduled based on 
// dims of start_info input
#include "colorizer.cl"
#include "cast_helpers.cl"

kernel void colored_retrace_starts(
	read_only image1d_t iS2_start_info,
	write_only image2d_t uc4_trace_image)
{
	short index = get_global_id(0);	// must be scheduled as 1D
	uint3 base_color = scatter_colorize(index);

	// initialize variables of arcs segment tracing loop for first iteration
	union l_conv coords;
	coords.i = read_imagei(iS2_start_info, index).lo;
	//only populated items in the array need to be processed
	if(!coords.l)
		return;

	write_imageui(uc4_trace_image, coords.i, (uint4)(256-base_color,-1) );
}