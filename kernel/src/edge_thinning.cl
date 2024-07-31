// Thins diagonal edge results from canny in the direction of the gradient so that intersection rejection is more feasible
#include "cast_helpers.cl"
#include "samplers.cl"

__kernel void edge_thinning(read_only image2d_t iC1_canny_image, write_only image2d_t iC1_thinned_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	char grad_ang = read_imagei(iC1_canny_image, coords).x;

	if(!grad_ang)	// only process populated cells
		return;
	
	// directions get reveresed on the 2nd edge thinning by changing the lookup table, there might be a better way to do this
#ifndef SECOND_THINNING
	const int2 offsets[] = {(int2)(1,0), (int2)(0,1), (int2)(-1,0), (int2)(0,-1), (int2)(1,0)};
#else
	const int2 offsets[] = {(int2)(-1,0), (int2)(0,-1), (int2)(1,0), (int2)(0,1), (int2)(-1,0)};
#endif//SECOND_THINNING

	int dir_idx = (uchar)grad_ang >> 6;	// which quadrant the gradient falls into
	union s_conv neighbors;	// populate face-sharing neighbor pixels
	neighbors.c.x = read_imagei(iC1_canny_image, clamped, coords + offsets[dir_idx]).x;
	neighbors.c.y = read_imagei(iC1_canny_image, clamped, coords + offsets[dir_idx + 1]).x;

	uchar2 diffs;
	if((neighbors.s & 0x0101) == 0x0101)	// both neighbors must be occupied to be elligible for thinning
	{
		diffs = abs(neighbors.c - grad_ang);
		if(all(diffs < (uchar)32))	// both relevant neighbors must have a similar direction
			return;	// current pixel is removed in thinning, skip writing out
	}
	// pixel was retained, write input value to output
	write_imagei(iC1_thinned_image, coords, grad_ang);
}