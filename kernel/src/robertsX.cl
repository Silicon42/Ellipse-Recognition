//intended mode is {REL, {-1,-1,0}}

// Roberts Cross gradient operator, less reads than Sobel/Scharr, math is mostly int add, less float mul
// should be faster with more fine resolution fidelity but more easily disturbed by noise

// [0] In	uc1_src_image: 1 channel greyscale on x component (UINT8)
// [1] Out	iS4_dst_image: 4 channel image of x and y gradient (INT16), angle 
//				(INT16), and gradient magnitude (UINT16)
__kernel void robertsX(read_only image2d_t uc1_src_image, write_only image2d_t iS2_dst_image)
{
	const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

	// Determine work item coordinate
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	int2 grad;
	grad.x = read_imageui(uc1_src_image, samp, coords - 1).x;
	grad.x -= read_imageui(uc1_src_image, samp, coords).x;
	grad.y = read_imageui(uc1_src_image, samp, coords + (int2)(0,-1)).x;
	grad.y -= read_imageui(uc1_src_image, samp, coords + (int2)(-1,0)).x;
	// correct for 45 deg offset, this is effectively a multiplication by 2*2
	// rotation matrix w/o the scale fixing component so that it stays an integer operation
	// since all calculated gradients get scaled the same way the scaling factor doesn't matter
	// range at this point is +/- 510 in each direction
	grad += (int2)(-grad.y, grad.x);

	short ang, mag;
	ang =0; // convert_short_sat_rte(atan2pi((float)grad.y, grad.x) * (1<<15));
	mag =-1;// convert_short_rte(fast_length(convert_float2(grad)));

	write_imagei(iS2_dst_image, coords, (int4)(grad, ang, mag));
}
