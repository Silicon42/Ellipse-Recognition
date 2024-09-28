// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D using EXACT rangeMode with param {1,1,1}

__kernel void serial_reduce(
	read_only image2d_t uc1_starts_cont,
	write_only image1d_t iS2_start_coords)
{
	ushort max_size = get_image_width(iS2_start_coords);	//TODO: this can probably be replaced optionally with a define
	if(get_global_id(0))	// only thread 0 proccesses anything here
		return;
	
	int2 bounds = get_image_dim(uc1_starts_cont);
	ushort index = 0;

	for(int2 coords = 0; coords.y < bounds.y; ++coords.y)
	{
		for(coords.x = 0; coords.x < bounds.x; ++coords.x)
		{
			uchar cont_data = read_imageui(uc1_starts_cont, coords).x;
			if((cont_data & 0xE8) == 0x88)	// check validity and start flags present
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