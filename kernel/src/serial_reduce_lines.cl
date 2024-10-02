// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D using EXACT rangeMode with param {1,1,1}
//TODO: rewrite this and serial_reduce to use multiple threads, can be done with each thread processing a chunk and saving a count
// then on a 2nd pass, compute the sum of all previous counts for each and then each serialized chunk can be appended
// to a single serialized list without conflicts
#include "cast_helpers.cl"

__kernel void serial_reduce_lines(
	read_only image1d_t iS2_start_coords,
	read_only image2d_t iC2_line_data,
	read_only image1d_t us1_line_counts,
	write_only image2d_t iS2_line_coords)
{
	int2 bounds = get_image_dim(iS2_line_coords);
	int max_size = bounds.x * bounds.y;	//TODO: this can probably be replaced optionally with a define

	uint index = 0;
	int2 coords;
	int seg_count;

	//for as many non-zero entries as iS2_start_coords has
	for(int i = 0; (i < max_size) && ((union l_conv)(coords = read_imagei(iS2_start_coords, i).lo)).l; ++i)
	{
		seg_count = read_imageui(us1_line_counts, i).x;
		for(int j = 0; j < seg_count; ++j)
		{
			//prevent overflow of the list
			if(index >= max_size)
			{
				printf("serial_reduce_lines(): maxed out at %u\n", index);
				return;
			}

			write_imagei(iS2_line_coords, convert_int2(((union s_conv)index).uc), (int4)(coords, 0, -1));
			++index;

			coords += read_imagei(iC2_line_data, coords).lo;
		}
	}

	printf("serial_reduce_lines(): found %u line segments, max was %u\n", index, max_size);
}