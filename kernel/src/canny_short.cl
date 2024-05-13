#define THRESH 16
//FIXME: ^ this ends up globally visible b/c of how OpenCL compiles files, see if compiling separately fixes this

int int2Dot(int2 a, int2 b)
{
	a *= b;
	return a.x + a.y;
}

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
	if ( grad.w <= THRESH)
		return;
	
	int2 n_grad = grad.lo*16 / (grad.w*9);	//just shy of 2 to 1 ratio w/o overflowing so that rounding stays bounded to +/- 1 even at the extremes

	// the logic here is too simple to work reliably on thin edges without first upsampling before edge finding since it doen't
	// check their direction and thus the opposed gradients of a 1 pixel wide line can annihilate each other, might also cause
	// problems at corners, which is more of what I'm worried about since I don't need 1 pixel wide line support
	int2 along;
	along.x = read_imagei(iS4_src_image, samp, coords - n_grad).w;
	along.y = read_imagei(iS4_src_image, samp, coords + n_grad).w;
	if(any(grad.w < along))	// non-max suppression
		return;
	
/*	// this doesn't work to fix it
	int2 along1, along2;
	along1 = read_imagei(iS4_src_image, samp, coords - n_grad).lo;
	along2 = read_imagei(iS4_src_image, samp, coords + n_grad).lo;

	if(int2Dot(along1, grad.lo) > grad.w || int2Dot(along2, grad.lo) > grad.w)
		return;
*/
	// compress to 7-bit angle and set occupancy flag
	write_imagei(iC1_dst_image, coords, (grad.z >> 8) | 1);
}