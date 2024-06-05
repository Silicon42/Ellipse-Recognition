// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D and should use SINGLE rangeMode

union ui2_conv {
	int2 i;
	uint2 u;
};

__kernel void serial_reduce(read_only image2d_t uc1_src_image, write_only image1d_t us4_dst_image)
{
	if(get_global_id(0))	// only thread 0 proccesses anything here
		return;
	
	int2 bounds = get_image_dim(uc1_src_image);
	union ui2_conv coords;
	coords.i = 0;
	short index = 0;
	for(; coords.i.y < bounds.y; ++coords.i.y)
	{
		for(; coords.i.x < bounds.x; ++coords.i.x)
		{
			ushort value = read_imageui(uc1_src_image, coords.i).x;
			if(value)
			{
				write_imageui(us4_dst_image, index, (uint4)(coords.u, value, -1));
			}
		}
	}

}