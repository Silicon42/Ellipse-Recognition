// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D using EXACT rangeMode with param {1,1,1}
#include "cast_helpers.cl"

__kernel void serial_reduce_arcs(read_only image2d_t ui4_arc_segments, write_only image1d_t is2_start_coords)
{
	ushort max_size = get_image_width(is2_start_coords);	//TODO: this can probably be replaced optionally with a define
	
	uint out_index = 1;	//index 0 reserved for size

	int2 bounds = get_image_dim(ui4_arc_segments);
	ushort index = 0;

	for(int2 coords = 0; coords.y < bounds.y; ++coords.y)
	{
		for(coords.x = 0; coords.x < bounds.x; ++coords.x)
		{
			union l_conv value;
			value.ui = read_imageui(ui4_arc_segments, coords).lo;
			if(!value)	// check occupancy for current pixel
				continue;
			
			write_imagei(is2_start_coords, index, (int4)(coords, 0, -1));
			++index;
			if(index == max_size)	// prevent possibly attempting to write past the end of the image, which can freeze the pipeline
			{
				printf("serial_reduce_arcs(): maxed out at %u\n", --index);
				write_imagei(is2_start_coords, 0, (int4)(index, 0, 0, -1));	//write fill level
				return;
			}
		}
	}
	printf("serial_reduce_arcs(): found %u arcs\n", index);
	write_imagei(is2_start_coords, 0, (int4)(index, 0, 0, -1));	//write fill level
}