// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D using EXACT rangeMode with param {1,1,1}
#include "cast_helpers.cl"
#include "path_struct_defs.cl"

__kernel void serial_reduce_arcs(read_only image2d_t ui4_arc_segments, write_only image2d_t iS2_start_coords, write_only image1d_t us4_lengths)
{
	uint max_size = get_image_width(iS2_start_coords);	//TODO: this can probably be replaced optionally with a define
	
	union ui4_array indices;

	indices.ui4 = 0;

	int2 bounds = get_image_dim(ui4_arc_segments);

	for(int2 coords = 0; coords.y < bounds.y; ++coords.y)
	{
		for(coords.x = 0; coords.x < bounds.x; ++coords.x)
		{
			union arc_rw arc_raw;
			arc_raw.ui4 = read_imageui(ui4_arc_segments, coords);
			if(!arc_raw.ul2.y)	// check occupancy for current pixel, this could technically fail if the center was at exactly (0,0) but that's not likely to ever happen
				continue;

			// which turning direction an arc will be treated as,
			// cw corresponds to 2, flat to 1, and ccw to 0
			uchar dir = arc_raw.data.ccw_mult + 1;
			dir = arc_raw.data.is_flat ? 1 : dir;

			if(indices.arr[dir] >= max_size)	//prevent overflow of any of the three lists without preventing the others from filling
				continue;
			
			write_imagei(iS2_start_coords, (int2)(indices.arr[dir], dir), (int4)(coords, 0, -1));
			++indices.arr[dir];
		}
	}

	if(any(indices.ui4 == max_size))	// warn that we ran out of space for the arcs
		printf("serial_reduce_arcs(): maxed out\n");

	printf("serial_reduce_arcs(): found %u cw arcs, %u flat arcs, and %u ccw arcs\n", indices.ui4);
	write_imageui(us4_lengths, 0, indices.ui4);	//write fill level
}