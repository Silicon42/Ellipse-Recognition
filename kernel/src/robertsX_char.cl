#define THRESH 80
#ifndef double	// fallback for devices without double support
// it's not super critical to function that this be a double in this file but it does prevent a rounding error
#define double float
//TODO: need to figure out how to warn the user that this has happened since it can effect later files and
// prevents the compiler from erroring out on it
#endif//double

// Roberts Cross gradient operator, less reads than Sobel/Scharr, math is mostly int add, less float mul
// should be faster with more fine resolution fidelity but more easily disturbed by noise

// [0] In	uc1_src_image: 1 channel greyscale on x component (UINT8)
// [1] Out	uc2_grad_image: 2 channel image of x and y gradient (INT8)
__kernel void robertsX_char(read_only image2d_t uc1_src_image, write_only image2d_t uc2_grad_image)
{
	// Determine work item coordinate
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	// gradient components relative to +45 degree offset for speed
	int2 grad;
	grad.x = read_imageui(uc1_src_image, coords + 1).x;
	grad.x -= read_imageui(uc1_src_image, coords).x;
	grad.y = read_imageui(uc1_src_image, coords + (int2)(0,1)).x;
	grad.y -= read_imageui(uc1_src_image, coords + (int2)(1,0)).x;

	// the gradient magnitude squared
	int2 grad2 = grad*grad;
	uint mag2;
	mag2 = grad2.x + grad2.y;
	
	// if the magnitude squared of the gradient doesn't meet the minimum threshold, no further processing needed
	if (mag2 < THRESH)
		return;

	uint mag = native_sqrt((float)mag2) * M_SQRT1_2_F;	// this is done to squeeze range into a uchar range

	float f_ang;
	if(grad.x)	// this is a fix so that we can still use the -cl-fast-relaxed-math compiler arg without divide by zero issues
		f_ang = atan2pi((double)grad.y, (double)grad.x);	// needs to be double if possible or else rounding inaccuracies sneak in
	else
		f_ang = (grad.y < 0) ? -0.5f : 0.5f;
	
	uchar ang = (char)floor(f_ang * 128) + 32;		// convert to fixed precision and correct for 45 deg offset

	write_imageui(uc2_grad_image, coords, (uint4)(ang, mag, 0, -1));
}
