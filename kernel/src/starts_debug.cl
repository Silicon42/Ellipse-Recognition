// Debug kernel that highligts starts and forced ends from find_segment_starts and edge pixels removed since edge thinning
// for visualization purposes

__kernel void starts_debug(read_only image2d_t iC1_thin, read_only image2d_t uc1_seg_start, write_only image2d_t uc4_dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	if(!read_imagei(iC1_thin, coords).x)
		return;
	
	uint4 color = (uint4)(0,0,0,-1);	//pixel starts full black
	char starts_data = read_imageui(uc1_seg_start, coords).x;

	if(!starts_data)	// if pixel got rejected
		color.x = -1;	// pixel set to full red
	else
		color.z = -1;	// all others set to full blue channel
	if(starts_data & 0x40)	// if pixel was a forced end
		color.x = -1;	// pixel set to magenta
	if(starts_data & 0x80)	// if pixel was a start
		color.y = -1;	// pixel set to cyan

	write_imageui(uc4_dst_image, coords, color);
}