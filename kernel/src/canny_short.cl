#define THRESH 12
//FIXME: ^ this ends up globally visible b/c of how OpenCL compiles files, see if compiling separately fixes this

// Alternate Canny function that expects shorts instead of floats
// [0] In	iS4_src_image: 4 channel image of x and y gradient (INT16), angle (INT16),
//				and gradient magnitude (INT16)
// [1] Out	iC1_dst_image: 1 channel signed 7-bit angle with 1 bit occupancy flag for
//				pixels meeting the gradient threshold and passing non-max suppression
//NOTE: Doesn't implement the hysteresis portion since that is inherently a very 
// serial operation, blurring and gradient computation is assumed to be already applied
__kernel void canny_short(read_only image2d_t iS4_src_image, write_only image2d_t iC1_dst_image)
{
	const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int4 grad = read_imagei(iS4_src_image, coords);

	// if the strength of the gradient doesn't meet the minimum requirement, no further processing needed
	if (THRESH > grad.w)
		return;
	
	int2 n_grad = grad.lo*64 / (grad.w*33);	//just shy of 2 to 1 ratio w/o overflowing so that rounding stays bounded to +/- 1 even at the extremes
	int2 along;
	along.lo = read_imagei(iS4_src_image, samp, coords - n_grad).w;
	along.hi = read_imagei(iS4_src_image, samp, coords + n_grad).w;
	if(any(along > grad.w))	// non-max suppression
		return;
	
	//grad.w = 0xffff;//debug modification so I can see the results
	//grad.lo /= grad.w;

	// compress to 7-bit angle and set occupancy flag
	write_imagei(iC1_dst_image, coords, (grad.z >> 8) | 1);
}