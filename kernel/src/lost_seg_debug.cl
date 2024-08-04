kernel void lost_seg_debug(read_only image2d_t uc4_retrace, read_only image2d_t uc1_cont_data, write_only image2d_t uc4_out)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uint4 color;
	uchar cont_data = read_imageui(uc1_cont_data, coords).x;
	switch(cont_data & 0x88)
	{
	case 0:		// non-edge pixel
		color = (uint4)(0,0,0,-1);
		break;
	case 0x88:	// valid thread-start
		color = -1;
		break;
	case 0x80:	// valid single-continuation non-start edge pixel
		color = read_imageui(uc4_retrace, coords);
		if(all(color == 0))
			color = (uint4)(-1,0,0,-1);
		break;
	default:	// multiple-continuation edge pixel
		color = (uint4)(-1,-1,0,-1);
	}
	
	write_imageui(uc4_out, coords, color);
}