// used to convert a single angle channel to a colorized version for better semantic viewing
// alternatively used to provide a unique color to individual "angle" items
// returns a color based on the angle on a color wheel as 3 0.0f to 1.0f values
// input angle is in half revolutions and should be in range +/- 1.0 for best accuracy
float3 index_colorize(int index)
{
	float3 garbage;	// required for modf() function call but I have no use for the integer part
	float3 angles = (float3)(0.0f, 0.66666667f, -0.66666667f);
	angles += angles;

	return fma(sinpi(modf(angles, &garbage)), 0.5, 0.5);
}

kernel void colored_retrace(read_only image1d_t us2_start_info, read_only image2d_t us4_path_image, write_only image2d_t uc4_trace_image)
{
	short index = get_global_id(0);	// must be scheduled as 1D
	float3 base_color = index_colorize(index);
	const int2 offsets[] = {(int2)(0,1),(int2)(-1,1),(int2)(-1,0),(int2)-1,(int2)(0,-1),(int2)(1,-1),(int2)(1,0),(int2)1};
	// initialize variables of arcs segment tracing loop for first iteration
	union l_i2 coords;
	union ul2_ui4 path;
	coords.ui = read_imageui(us2_start_info, index).lo;
	while(1)
	{
		path.ui = read_imageui(us4_path_image, coords.i);
		uchar path_len = path.uc.s8 & 0x3F;
		if(path_len == 0)	// if length indicates 0 here, then there was no further processing on this work item
			return;
		if(path_len > ACCUM_STRUCT_LEN2)	//TODO: indicating continuation might not be neccessary, for now it's left in though
			path_len = ACCUM_STRUCT_LEN2;
		
		for(; path_len > 0; --path_len)
		{
			write_imageui(uc4_trace_image, coords.i, base_color);
			if(path_len == ACCUM_STRUCT_LEN1)
				path.ul.x |= path.ul.y & -64L;
			
			coords.i += offsets[path.ul.x & 7];
			path.ul.x >>= 3;
		}


	}
}