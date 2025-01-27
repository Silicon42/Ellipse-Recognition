// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D using EXACT rangeMode with param {1,1,1}
//TODO: replace this kernel with a proper reduction once a working proof of concept is done
#include "cast_helpers.cl"

__kernel void serial_reduce_arc_seeds(
	read_only image2d_t us1_seg_in_arc,
//	write_only image1d_t us1_length,	//is actually length -1 to not possibly overflow
	write_only image1d_t is2_arc_coords)
{
	ushort max_size = get_image_width(is2_arc_coords);	//TODO: this can probably be replaced optionally with a define
	if(get_global_id(0))	// only thread 0 proccesses anything here
		return;
	
	int2 bounds = get_image_dim(us1_seg_in_arc);
	ushort index = 0;

	for(int2 coords = 0; coords.y < bounds.y; ++coords.y)
	{
		for(coords.x = 0; coords.x < bounds.x; ++coords.x)
		{
			uchar seg_cnt = read_imageui(us1_seg_in_arc, coords).x;
			if(seg_cnt >= 4)
			{
				write_imagei(is2_arc_coords, index, (int4)(coords, 0, -1));
				++index;
				if(index == max_size)	// prevent possibly attempting to write past the end of the image, which can freeze the pipeline
				{
					printf("serial_reduce_arc_seeds(): maxed out at %u\n", index);
					return;
				}
			}
		}
	}
	printf("serial_reduce_arc_seeds(): max index was %u\n", index);
}