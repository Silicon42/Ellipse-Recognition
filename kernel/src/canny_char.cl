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
__kernel void canny_char(read_only image2d_t iS4_src_image, write_only image2d_t iC1_dst_image)
{
	const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	const int2 offsets[4] = {(int2)(1,0),(int2)(1,1),(int2)(0,1),(int2)(-1,1)};

	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 grad = read_imagei(iS4_src_image, coords).lo;

	// if the strength of the gradient doesn't meet the minimum requirement, no further processing needed
	if ( grad.y <= THRESH)
		return;
	
	uchar along_idx = (grad.x >> 5) & 3;

	int2 along;
	// check against gradient direction
	along = read_imagei(iS4_src_image, samp, coords - offsets[along_idx]).lo;
	// only compare similarly aligned gradient magnitudes
	if(abs(along.x - grad.x) < 64)
	{
		if(any(grad.y < along.y))	// non-max suppression
			return;
		if(any(grad.y == along.y) && ((coords.x + coords.y) & 1))	// constant gradient edge case mitigation
			return;
	}
	// check along gradient direction
	along = read_imagei(iS4_src_image, samp, coords + offsets[along_idx]).lo;
	// only compare similarly aligned gradient magnitudes
	if(abs(along.x - grad.x) < 64)
	{
		if(any(grad.y < along.y))	// non-max suppression
			return;
		if(any(grad.y == along.y) && ((coords.x + coords.y) & 1))	// constant gradient edge case mitigation
			return;
	}

	// set occupancy flag
	write_imagei(iC1_dst_image, coords, grad.x | 1);
}