// displays left and right link direction of input via a 3x bigger output image

__kernel void link_debug(read_only image2d_t uc1_cont, write_only image2d_t iC4_debug_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uchar cont_data = read_imageui(uc1_cont, coords).x;

	if(!cont_data)	// only process populated cells
		return;
	
	coords = coords * 3 + 1;
	const int2 offsets[] = {(int2)(1,0),(int2)1,(int2)(0,1),(int2)(-1,1),(int2)(-1,0),(int2)-1,(int2)(0,-1),(int2)(1,-1)};

	write_imagei(iC4_debug_image, coords, (int4)(-1,0,0,-1));	// red for pixel itself
	if(cont_data & 16)
		write_imagei(iC4_debug_image, coords + offsets[cont_data >> 5], (int4)(0,-1,0,-1));	// green for left pixel
	if(cont_data & 8)
		write_imagei(iC4_debug_image, coords + offsets[cont_data & 7], (int4)(0,0,-1,-1));	// blue for right pixel
}