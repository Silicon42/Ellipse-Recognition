
#include "colorizer.cl"
#include "cast_helpers.cl"
#include "path_struct_defs.cl"

kernel void colored_retrace(read_only image1d_t us2_start_info, read_only image2d_t ui4_path_image, write_only image2d_t uc4_trace_image)
{
	short index = get_global_id(0);	// must be scheduled as 1D
	uint3 base_color = scatter_colorize(index);
	const int2 offsets[] = {(int2)(0,1),(int2)(-1,1),(int2)(-1,0),-1,(int2)(0,-1),(int2)(1,-1),(int2)(1,0),1};
	// initialize variables of arcs segment tracing loop for first iteration
	union l_conv coords;
	ulong2 path;
	coords.ui = read_imageui(us2_start_info, index).lo;
	if(!coords.l)
		return;
	//uchar is_extended = 1;
	while(1)
	{
		uchar path_len = read_data_accum(&path, ui4_path_image, coords.i);
		if(path_len == 0)	// if length indicates 0 here, then there was no further processing on this work item
			return;
		//mark as start/restart
		write_imageui(uc4_trace_image, coords.i, (uint4)(base_color+32,-1) );

		//is_extended = path_len > ACCUM_STRUCT_LEN2;
		if(path_len > ACCUM_STRUCT_LEN2)	//TODO: indicating continuation might not be neccessary, for now it's left in though
			path_len = ACCUM_STRUCT_LEN2;
		
		for(uchar i = 0; i < path_len; ++i)
		{
			if(i == ACCUM_STRUCT_LEN1)
				path.x = path.y;

			coords.i += offsets[path.x & 7];
			write_imageui(uc4_trace_image, coords.i, (uint4)(base_color/2, -1));
			path.x >>= 3;
		}
	}
}