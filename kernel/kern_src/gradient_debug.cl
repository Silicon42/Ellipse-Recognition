// displays occupancy and gradient direction of input via a 3x bigger output image

__kernel void gradient_debug(
	read_only image2d_t ic1_grad_image,
	write_only image2d_t uc4_debug_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uchar grad_ang = read_imagei(ic1_grad_image, coords).x;

	if(!grad_ang)	// only process populated cells
		return;
	
	coords = coords * 3 + 1;
	const int2 offsets[] = {(int2)(1,0),(int2)1,(int2)(0,1),(int2)(-1,1),(int2)(-1,0),(int2)-1,(int2)(0,-1),(int2)(1,-1),
							(int2)(1,0),(int2)1};
	int dir_idx = (uchar)(grad_ang + 16) >> 5;	// which octant the gradient falls into

	write_imageui(uc4_debug_image, coords, -1);
	write_imageui(uc4_debug_image, coords + offsets[dir_idx], (uint4)(-1,0,0,-1));
	write_imageui(uc4_debug_image, coords + offsets[dir_idx + 2], (uint4)(0,-1,0,-1));
}