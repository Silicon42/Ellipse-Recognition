
#define THRESH 9

#ifndef double	// fallback for devices without double support
// it's not super critical to function that this be a double in this file but it does prevent a rounding error
#define double float
#define DOUBLE_NA
#warning "No double support on the requested platform, falling back to float."
//TODO: need to figure out how to warn the user that this has happened since it can effect later files and
// prevents the compiler from erroring out on it
#endif//double*/

//TODO: check if recycling variables via macros reduces private mem usage
//TODO: check if image reads can be made more simultaneous to boost performance
// possibly by repeating shifted versions of the image on other channels or if the
// extra processing that requires is too much.
//TODO: see if converting to short integer operations improves speed
//TODO: consider adding "Magic Kernel Sharp" to preserve higher edge resolution
// as detailed here: https://johncostella.com/edgedetect/

// [0] In	fc1_src_image: 1 channel greyscale on x component (UNORM)
// [1] Out	uc2_grad: 2 channels, angle + gradient magnitude
__kernel void scharr3_char(read_only image2d_t fc1_src_image, write_only image2d_t uc2_grad)
{
	const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	// Determine work item coordinate
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	// itermediate differences for horizontal/vertical scharr shared operations
	float diag;
	float2 grad;

//TODO: check if the ordering of these reads effects performance or if the compiler is smart enough that it doesn't matter
	// calculate intermediate differences from raw pixels
	diag  = read_imagef(fc1_src_image, samp, coords + 1).x;
	diag -= read_imagef(fc1_src_image, samp, coords - 1).x;
	grad.x  = read_imagef(fc1_src_image, samp, coords + (int2)(1,0)).x;
	grad.x -= read_imagef(fc1_src_image, samp, coords - (int2)(1,0)).x;
	grad.y  = read_imagef(fc1_src_image, samp, coords + (int2)(0,1)).x;
	grad.y -= read_imagef(fc1_src_image, samp, coords - (int2)(0,1)).x;
	grad = fma(grad, 3.44680851f, diag);

	// recycle diag for the other diagonal difference
	diag  = read_imagef(fc1_src_image, samp, coords + (int2)(1,-1)).x;
	diag -= read_imagef(fc1_src_image, samp, coords - (int2)(1,-1)).x;
	grad += (float2)(diag,-diag);

	float f_ang, f_mag;
	uchar ang, mag;
	// normalize magnitude such that we get the most resolution out of a byte possible,
	// all below small threshold end up negative and saturate to 0, maintaining best comparison resolution available
	f_mag = fma(fast_length(grad), 38, -THRESH);	//max grad magnitude: sqrt(48.94182888184698958804889090086) == ~7
	mag = convert_uchar_sat_rte(f_mag);
	// if the magnitude of the gradient did't meet the minimum threshold, no further processing needed
	if (!mag)
		return;
	// normalize angle such that conversion to int types wraps properly
	//if(grad.x)	// this is a fix so that we can still use the -cl-fast-relaxed-math compiler arg without divide by zero issues
		f_ang = atan2pi((double)grad.y, (double)grad.x);	// needs to be double if possible or else rounding inaccuracies sneak in
	//else
	//	f_ang = (grad.y < 0) ? -0.5f : 0.5f;
	
	ang = (char)floor(f_ang * 128);

	write_imageui(uc2_grad, coords, (uint4)(ang, mag, 0, -1));
}