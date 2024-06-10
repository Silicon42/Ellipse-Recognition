// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D and should use SINGLE rangeMode

__kernel void serial_reduce(read_only image2d_t uc1_src_image, write_only image1d_t us4_dst_image)
{
	if(get_global_id(0))	// only thread 0 proccesses anything here
		return;
	
	int2 bounds = get_image_dim(uc1_src_image);
	short index = 0;

	for(int y_coord = 0; y_coord < bounds.y; ++y_coord)
	{
		for(int x_coord = 0; x_coord < bounds.x; ++x_coord)
		{
			uchar value = read_imageui(uc1_src_image, (int2)(x_coord, y_coord)).x;
			if(value)
			{
				write_imageui(us4_dst_image, index, (uint4)(x_coord , y_coord, value, -1));
				++index;
			}
		}
	}

}