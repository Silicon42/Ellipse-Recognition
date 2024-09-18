// redraws the initial starts after colored_retrace since some may have gotten
// overwritten by the end pixel of other threads, should be scheduled based on 
// dims of start_info input
#include "colorizer.cl"
#include "cast_helpers.cl"

kernel void colored_retrace_starts(
	read_only image1d_t iS2_start_info,
	read_only image2d_t ui4_path_image,
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

	// if location wasn't written to, which should include at least 1 bit set in the lowest 6 bits,
	// then the read should return 0 and there is no further processing on this work item
	if(!read_imageui(ui4_path_image, coords.i).x)
		return;

	write_imageui(uc4_trace_image, coords.i, (uint4)(base_color+32,-1) );
}