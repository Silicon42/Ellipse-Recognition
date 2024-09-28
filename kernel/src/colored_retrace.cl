// redraws the arc segments from the stored path accumulators with psuedo-random
// colors to visualize the processing of individual work items, should be scheduled
// based on dims of start_info input
#include "colorizer.cl"
#include "cast_helpers.cl"
#include "bresenham_line.cl"
#include "arc_data.cl"

kernel void colored_retrace(
	read_only image2d_t uc1_cont_info,
	read_only image1d_t iS2_start_info,
	read_only image2d_t ui4_arc_data,
	write_only image2d_t uc4_trace_image)
{
//	struct arc_data arc = read_arc_data(ui4_arc_data, coords);
//	if(arc.len)// && !(arc.flags & IS_NOT_END))


	// initialize variables of arcs segment tracing loop for first iteration
	int index = get_global_id(0);	// must be scheduled as 1D
	int2 coords = read_imagei(iS2_start_info, index).lo;
	//only populated items in the array need to be processed
	if(!(coords.x || coords.y))
		return;

	int2 end_coords, mid_coords;
	uint3 base_color = scatter_colorize(index);
	struct arc_data arc;
	do
	{
		arc = read_arc_data(ui4_arc_data, coords);
		if(arc.len < 1)	// if length indicates 0 here, then there was no further processing on this work item
		{
			printf("illegal arc length at (%i, %i)\n", coords);
			write_imageui(uc4_trace_image, coords, (uint4)(-1, 0, 0, -1));
			return;
		}
		if(!(arc.offset_end.x || arc.offset_end.y))
		{
			printf("illegal arc endpoint at (%i, %i)\n", coords);
			write_imageui(uc4_trace_image, coords, (uint4)(-1, 0, 0, -1));
			return;
		}
		if(!(arc.offset_mid.x || arc.offset_mid.y))
		{
			printf("illegal arc midpoint at (%i, %i)\n", coords);
			write_imageui(uc4_trace_image, coords, (uint4)(-1, 0, 0, -1));
			return;
		}


		end_coords = coords + convert_int2(arc.offset_end);
		mid_coords = coords + convert_int2(arc.offset_mid);
		draw_line(coords, mid_coords, (uint4)(base_color, -1), uc4_trace_image);
		draw_line(mid_coords, end_coords, (uint4)(base_color, -1), uc4_trace_image);
		write_imageui(uc4_trace_image, mid_coords, (uint4)(base_color + 80, -1));

		//mark as start/restart
		write_imageui(uc4_trace_image, coords, (uint4)(256-base_color, -1));
		coords = end_coords;
/*
		//is_extended = path_len > ACCUM_STRUCT_LEN2;
		if(path_len > ACCUM_STRUCT_LEN2)	//TODO: indicating continuation might not be neccessary, for now it's left in though
			path_len = ACCUM_STRUCT_LEN2;
		
		for(uchar i = 0; i < path_len; ++i)
		{
			if(i == ACCUM_STRUCT_LEN1)
				path.x = path.y >> 6;

			coords.i += offsets[path.x & 7];
			write_imageui(uc4_trace_image, coords.i, (uint4)(base_color/2, -1));
			path.x >>= 3;
		}*/
	} while(arc.flags & IS_NOT_END);
}