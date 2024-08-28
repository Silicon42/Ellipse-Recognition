// reduce very sparse 2D info to compact 1D
// This might get replaced with a simple hash and retry on collision method later so that it's not a serial bottleneck
//NOTE: must be scheduled as 1D and should use SINGLE rangeMode

__kernel void serial_reduce(read_only image2d_t ui4_arc_segments, write_only image1d_t is2_start_info)
{
	ushort max_size = get_image_width(is2_start_info);	//TODO: this can probably be replaced optionally with a define
	if(get_global_id(0))	// only thread 0 proccesses anything here
		return;
	
	int2 bounds = get_image_dim(ui4_arc_segments);
	ushort index = 0;

	for(int y_coord = 0; y_coord < bounds.y; ++y_coord)
	{
		for(int x_coord = 0; x_coord < bounds.x; ++x_coord)
		{
			uchar value = read_imageui(ui4_arc_segments, (int2)(x_coord, y_coord)).x;
			if((value & 0x88) == 0x88)	// check validity and start flags present
			{
				write_imagei(is2_start_info, index, (int4)(x_coord , y_coord, 0, -1));
				++index;
				if(index == max_size)	// prevent possibly attempting to write past the end of the image, which can freeze the pipeline
				{
					printf("serial_reduce(): maxed out at %u\n", index);
					//write_imagei(is2_start_info, 0, max_size);
					return;
				}
			}
		}
	}
	printf("serial_reduce(): max index was %u\n", index);
	// Index 0 is reserved for the length of the occupied portion of the array
	//write_imageui(is2_start_info, 0, index);
}