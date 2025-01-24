// debugging kernel for figuring out which segments are actually being processed and which are being lost

kernel void lost_seg_debug(
	read_only image2d_t uc4_retrace,
	read_only image2d_t uc1_cont_data,
	write_only image2d_t uc4_out)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uint4 color = read_imageui(uc4_retrace, coords);
	// any other remaining pixels got rejected or unintentionally lost and are decided based on the results of cont_data
	if(!color.w)
	{
		uchar cont_data = read_imageui(uc1_cont_data, coords).x;
		switch(cont_data & 0x88)
		{
		case 0:		// non-edge pixel
			color = (uint4)(0,0,0,-1);	// full black
			break;
		case 0x88:	// valid thread-start, start flag and right continuation flag set
			color = -1;	// full white
			break;
		case 0x08:	// valid right-continuation non-start edge pixel
			color = (uint4)(-1,0,0,-1);	// full red
			break;
		default:	// start with no right continuation
			color = (uint4)(-1,-1,0,-1);	// full yellow
		}
	}
	
	write_imageui(uc4_out, coords, color);
}