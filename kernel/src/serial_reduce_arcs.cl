// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D using EXACT rangeMode with param {1,1,1}
//TODO: rewrite this and serial_reduce to use multiple threads, can be done with each thread processing a chunk and saving a count
// then on a 2nd pass, compute the sum of all previous counts for each and then each serialized chunk can be appended
// to a single serialized list without conflicts
#include "cast_helpers.cl"
#include "arc_data.cl"

__kernel void serial_reduce_arcs(
	read_only image1d_t iS2_start_coords,
	read_only image2d_t ui4_arc_data,
	write_only image2d_t iS2_arc_coords,
	write_only image1d_t us2_lengths)
{
	printf("serial_reduce_arcs entry\n");
	int max_size = get_image_width(iS2_start_coords);	//TODO: this can probably be replaced optionally with a define

	struct arc_data arc;
	uint indices[2] = {0};
	int2 coords;

	//for as many non-zero entries as iS2_start_coords has
	for(int i = 0; (i < max_size) && ((union l_conv)(coords = read_imagei(iS2_start_coords, i).lo)).l; ++i)
	{
		do	// read each arc in the processing chain until we come to one flagged with IS_END
		{
			arc = read_arc_data(ui4_arc_data, coords);

			// which turning direction an arc will be treated primarily as
			uchar dir = (arc.flags & IS_CW) >> 1;

			//prevent overflow of any of the two lists without preventing the others from filling
			if(indices[dir] < max_size)
				write_imagei(iS2_arc_coords, (int2)(indices[dir], dir), (int4)(coords, 0, -1));
			++indices[dir];

			// skip the next section if not being treated as both cw and ccw (only flat should have this flag set)
			if(!(arc.flags & IS_BOTH_HANDED))
				continue;
			
			if(indices[!dir] < max_size)
				write_imagei(iS2_arc_coords, (int2)(indices[!dir], !dir), (int4)(coords, 0, -1));
			++indices[!dir];
		}while(arc.flags & IS_NOT_END);
	}

	uint2 lengths = (uint2)(indices[0], indices[1]);
	printf("serial_reduce_arcs(): found %u cw arcs, and %u ccw arcs, max was %u\n", lengths, max_size);
	lengths = max(lengths, max_size);
	
	write_imageui(us2_lengths, 0, (uint4)(lengths, 0,0));	//write fill level
}