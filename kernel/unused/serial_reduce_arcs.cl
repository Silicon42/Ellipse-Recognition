//BAD EARLY VERSION may revisit if I fix the data structure to include a flag for if a segment was the first in the chain
// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D using EXACT rangeMode with param {1,1,1}
#include "cast_helpers.cl"
#include "path_struct_defs.cl"

__kernel void serial_reduce_arcs(read_only image1d_t us2_start_coords, read_only image2d_t ui4_arc_segments, write_only image1d_t is2_start_coords)
{
	ushort max_size = get_image_width(is2_start_coords);	//TODO: this can probably be replaced optionally with a define
	
	uint out_index = 1;	//index 0 reserved for size

	union l_conv coords;
	union arc_rw arc_data;
	for(ushort in_index = 0; ; ++in_index)
	{
		// initialize variables of arcs segment tracing loop for first iteration
		coords.ui = read_imageui(us2_start_coords, in_index).lo;
		if(!coords.l)	//loop until we read a null value, signifying the end of segment starts data
			return;
		
		for(; out_index < max_size; ++out_index)
		{
			arc_data.ui4 = read_imageui(ui4_arc_segments, coords.i);
			// check occupancy, some starts might be empty due to being rejected as short without supporting segments
			if(!arc_data.ul2.y)
				break;
		
			write_imagei(is2_start_coords, out_index, (int4)(coords.i, 0, -1));
			//this doesn't work and can cause a loop because there's no check that an arc belonged to a single processing thread
			coords.i += convert_int2(arc_data.data.offset_end);
		}
	}
	printf("serial_reduce_arcs(): found %u arcs\n", out_index);
	write_imagei(is2_start_coords, 0, (int4)(out_index, 0, 0, -1));	//write fill level
}