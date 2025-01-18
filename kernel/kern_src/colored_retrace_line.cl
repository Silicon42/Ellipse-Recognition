// redraws the arc segments from the stored path accumulators with psuedo-random
// colors to visualize the processing of individual work items, should be scheduled
// based on dims of start_info input
#include "colorizer.cl"
#include "cast_helpers.cl"
#include "bresenham_line.cl"

kernel void colored_retrace_line(
	read_only image2d_t uc1_cont_info,
	read_only image1d_t is2_start_info,
	read_only image2d_t ic2_line_data,
	read_only image1d_t us1_line_counts,
	write_only image2d_t uc4_trace_image)
{
	// initialize variables of arcs segment tracing loop for first iteration
	int index = get_global_id(0);	// must be scheduled as 1D
	int seg_count = read_imageui(us1_line_counts, index).x;
	//only populated items in the array need to be processed
	if(!seg_count)
		return;
	
	int2 coords = read_imagei(is2_start_info, index).lo;
	int2 end_offset, end_coords;
	uint3 base_color = scatter_colorize(index);

	for(int i = 0; i < seg_count; ++i)
	{
		end_offset = read_imagei(ic2_line_data, coords).lo;
		if(!(end_offset.x || end_offset.y))
		{
			printf("illegal arc endpoint at (%i, %i)\n", coords);
			write_imageui(uc4_trace_image, coords, (uint4)(-1, 0, 0, -1));
			return;
		}

		end_coords = coords + end_offset;
		draw_line(coords, end_coords, (uint4)(base_color, -1), uc4_trace_image);
		//mark as start/restart
		write_imageui(uc4_trace_image, coords, (uint4)(256-base_color, -1));
		coords = end_coords;
	}
}