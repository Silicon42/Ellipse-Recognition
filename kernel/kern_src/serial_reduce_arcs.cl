// reduce very sparse 2D info (conic arcs) to compact form
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D using EXACT rangeMode with param {1,1,1}
//TODO: replace this kernel with a proper reduction once a working proof of concept is done
#include "arc_data.cl_h"

__kernel void serial_reduce_arcs(
	read_only image2d_t uc1_dir_cnt,
	write_only image2d_t is2_arc_coords)
{
	ushort max_size = get_image_width(is2_arc_coords);	//TODO: this can probably be replaced optionally with a define
	if(get_global_id(0))	// only thread 0 proccesses anything here
		return;
	
	int2 bounds = get_image_dim(uc1_dir_cnt);
	int index[2] = {0};

	for(int2 coords = 0; coords.y < bounds.y; ++coords.y)
	{
		for(coords.x = 0; coords.x < bounds.x; ++coords.x)
		{
			if(all(coords == 0))	//prevent arcs at (0,0) from possibly writing, since that currently signifies the end of the list
				continue;	//TODO: fix it so that arcs at (0,0) don't cause problems
			uchar dir_cnt = read_imageui(uc1_dir_cnt, coords).x;
			// if there isn't enough points for a stable solution, skip adding it as an arc to start processing from
			if(!(dir_cnt & SEG_CNT_MASK))
				continue;

			uchar is_ccw = dir_cnt >> DIR_SHIFT;
			if(index[is_ccw] == max_size)	// prevent possibly attempting to write past the end of the image, which can freeze the pipeline
			{
				printf("serial_reduce_arcs(): maxed out at %u [%u]\n", max_size, is_ccw);
				continue;
			}
			//else
			write_imagei(is2_arc_coords, (int2)(index[is_ccw], is_ccw), (int4)(coords,0,0));
			++index[is_ccw];
		}
	}
	printf("serial_reduce_arcs(): max indices were cw: %u, ccw: %u\n", index[0], index[1]);
}