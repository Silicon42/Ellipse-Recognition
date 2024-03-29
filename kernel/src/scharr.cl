const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
//TODO: check if recycling variables via macros reduces private mem usage
//TODO: check if image reads can be made more simultaneous to boost performance
// possibly by repeating shifted versions of the image on other channels or if the
// extra processing that requires is too much.
//TODO: see if converting to short integer operations improves speed
//TODO: consider adding "Magic Kernel Sharp" to preserve higher edge resolution
// as detailed here: https://johncostella.com/edgedetect/

// [0] In	fc1_src_image: 1 channel greyscale on x component (UNORM)
// [1] Out	fF4_dst_image: 4 channel image of x and y gradient (SFLOAT), angle 
//				(SNORM), and gradient magnitude (UFLOAT)
__kernel void scharr(read_only image2d_t fc1_src_image, write_only image2d_t fF4_dst_image)
{
	// Determine work item coordinate
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	// itermediate differences for horizontal/vertical scharr shared operations
	float diag, ang, mag;
	float2 grad;

	// calculate intermediate differences from raw pixels
	diag = read_imagef(fc1_src_image, sampler, coords - 1).x;
	diag -= read_imagef(fc1_src_image, sampler, coords + 1).x;
	grad.x = read_imagef(fc1_src_image, sampler, coords - (int2)(1,0)).x;
	grad.x -= read_imagef(fc1_src_image, sampler, coords + (int2)(1,0)).x;
	grad.y = read_imagef(fc1_src_image, sampler, coords - (int2)(0,1)).x;
	grad.y -= read_imagef(fc1_src_image, sampler, coords + (int2)(0,1)).x;
	grad = fma(grad, 3.44680851f, diag);

	// recycle diag for the other diagonal difference
	diag = read_imagef(fc1_src_image, sampler, coords - (int2)(1,-1)).x;
	diag -= read_imagef(fc1_src_image, sampler, coords + (int2)(1,-1)).x;
	grad += (float2)(diag,-diag);

	// re-normalize values. Only necessary if trying to view image for debugging
	//grad = fma(grad, 0.091796875f, 0.5f);

	// normalize angle such that conversion to int types wraps properly
	ang = atan2pi(grad.y, grad.x);
	mag = length(grad);

	write_imagef(fF4_dst_image, coords, (float4)(grad, ang, mag));
}