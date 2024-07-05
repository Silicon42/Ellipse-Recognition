//TODO: check if recycling variables via macros reduces private mem usage
//TODO: check if image reads can be made more simultaneous to boost performance
// possibly by repeating shifted versions of the image on other channels or if the
// extra processing that requires is too much.
//TODO: see if converting to short integer operations improves speed
//TODO: consider adding "Magic Kernel Sharp" to preserve higher edge resolution
// as detailed here: https://johncostella.com/edgedetect/

// [0] In	fc1_src_image: 1 channel greyscale on x component (UNORM)
// [1] Out	iC2_dst_image: 2 channels, angle + gradient magnitude
__kernel void scharr3_char(read_only image2d_t fc1_src_image, write_only image2d_t iC2_dst_image)
{
	const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	// Determine work item coordinate
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	// itermediate differences for horizontal/vertical scharr shared operations
	float diag;
	float2 grad;

	// calculate intermediate differences from raw pixels
	diag = read_imagef(fc1_src_image, samp, coords - 1).x;
	diag -= read_imagef(fc1_src_image, samp, coords + 1).x;
	grad.x = read_imagef(fc1_src_image, samp, coords - (int2)(1,0)).x;
	grad.x -= read_imagef(fc1_src_image, samp, coords + (int2)(1,0)).x;
	grad.y = read_imagef(fc1_src_image, samp, coords - (int2)(0,1)).x;
	grad.y -= read_imagef(fc1_src_image, samp, coords + (int2)(0,1)).x;
	grad = fma(grad, 3.44680851f, diag);

	// recycle diag for the other diagonal difference
	diag = read_imagef(fc1_src_image, samp, coords - (int2)(1,-1)).x;
	diag -= read_imagef(fc1_src_image, samp, coords + (int2)(1,-1)).x;
	grad += (float2)(diag,-diag);

	char ang, mag;
	// normalize angle such that conversion to int types wraps properly
	ang = convert_char_sat_rte(atan2pi(grad.y, grad.x) * 128);
	mag = (char)convert_uchar_sat_rte(fast_length(grad) * 36.5f);	//max grad magnitude: sqrt(48.94182888184698958804889090086) == ~7

	write_imagei(iC2_dst_image, coords, (int4)(ang, mag, 0, -1));
}