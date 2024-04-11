
#define THRESH 16

// Alternate Canny function that expects shorts instead of floats
// [0] In	iS4_src_image: 4 channel image of x and y gradient (INT16), angle (INT16),
//				and gradient magnitude (INT16)
// [1] Out	iS4_dst_image: 4 channel masked iS4_src_image angle and magnitude transferred
//				directly and x and y gradient normalized for pixels 
//				meeting the gradient threshold and passing non-max suppression
//NOTE: Doesn't implement the hysteresis portion since that is inherently a very 
// serial operation, blurring and gradient computation is assumed to be already applied
__kernel void canny_short(read_only image2d_t iS4_src_image, write_only image2d_t iS4_dst_image)
{
	//TODO: check if it's cheaper to manually filter between the adjacent pixels since it would access half the pixels
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
	
	grad.w = 0xffff;//debug modification so I can see the results
	//grad.lo /= grad.w;
	write_imagei(iS4_dst_image, coords, grad);
}