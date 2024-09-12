// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D using EXACT rangeMode with param {1,1,1}

__kernel void serial_reduce(read_only image2d_t uc1_src_image, write_only image1d_t iS2_start_coords)
{
	ushort max_size = get_image_width(iS2_start_coords);	//TODO: this can probably be replaced optionally with a define
	if(get_global_id(0))	// only thread 0 proccesses anything here
		return;
	
	int2 bounds = get_image_dim(uc1_src_image);
	ushort index = 0;

	for(int2 coords = 0; coords.y < bounds.y; ++coords.y)
	{
		for(coords.x = 0; coords.x < bounds.x; ++coords.x)
		{
			uchar value = read_imageui(uc1_src_image, coords).x;
			if((value & 0x88) == 0x88)	// check validity and start flags present
			{
				write_imagei(iS2_start_coords, index, (int4)(coords, 0, -1));
				++index;
				if(index == max_size)	// prevent possibly attempting to write past the end of the image, which can freeze the pipeline
				{
					printf("serial_reduce(): maxed out at %u\n", index);
					return;
				}
			}
		}
	}
	printf("serial_reduce(): max index was %u\n", index);
}