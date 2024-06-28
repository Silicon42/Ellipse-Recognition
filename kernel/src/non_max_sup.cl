
const sampler_t clamped = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

// Alternate Canny function that expects chars instead of floats
// [0] In	uc2_grad_image: 4 channel image of x and y gradient (INT16), angle (INT16),
//				and gradient magnitude (INT16)
// [1] Out	iC1_dst_image: 1 channel signed 7-bit angle with 1 bit occupancy flag for
//				pixels meeting the gradient threshold and passing non-max suppression
//NOTE: Doesn't implement the hysteresis portion since that is inherently a very 
// serial operation, blurring and gradient computation is assumed to be already applied
__kernel void non_max_sup(read_only image2d_t uc2_grad_image, write_only image2d_t iC1_dst_image)
{
	const int2 offsets[4] = {(int2)(1,0),(int2)(1,1),(int2)(0,1),(int2)(-1,1)};

	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uint2 grad = read_imageui(uc2_grad_image, coords).lo;

	// if the strength of the gradient wasn't recorded, it didn't meet the minimum threshold, no further processing needed
	if(!grad.y)
		return;

	uchar dir_idx = ((grad.x + 16) >> 5) & 3;	// convert angle into binned index into offsets table

	// read pixels in and against the direction of the gradient to compare to
	uint4 along;
	along.lo = read_imageui(uc2_grad_image, clamped, coords + offsets[dir_idx]).lo;
	along.hi = read_imageui(uc2_grad_image, clamped, coords - offsets[dir_idx]).lo;

	// verify that angle of the pixels read is within +/- 45 degrees,
	// this allows for processing of thin lines and sharp corners correctly
	char2 is_angle_similar = abs((char)grad.x - convert_char2(along.even)) < (char)32;
	// mask magnitudes conditionally by if they were close in angle
	uint2 validated_mag = along.odd & convert_uint2(is_angle_similar);
	if(any(validated_mag > grad.y))	// non-max suppression
		return;
	if(any(validated_mag == grad.y) && ((coords.x + coords.y) & 1))	// constant gradient edge case mitigation, typically only in artificial images
		return;

	// set occupancy flag
	write_imagei(iC1_dst_image, coords, (char)grad.x | 1);
}