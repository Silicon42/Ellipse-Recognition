//intended mode is {REL, {1,1,0}}
//FIXME: VVVVV this ends up globally visible b/c of how OpenCL compiles files, see if compiling separately fixes this
#define THRESH 512

const sampler_t edge_clamp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// Roberts Cross gradient operator, less reads than Sobel/Scharr, math is mostly int add, less float mul
// should be faster with more fine resolution fidelity but more easily disturbed by noise

// [0] In	uc1_src_image: 1 channel greyscale on x component (UINT8)
// [1] Out	uc2_grad_image: 2 channel image of x and y gradient (INT8)
__kernel void robertsX_char(read_only image2d_t uc1_src_image, write_only image2d_t uc2_grad_image)
{
	// Determine work item coordinate
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	// gradient is left relative to +45 degree offset for speed
	int2 grad;
	grad.x = read_imageui(uc1_src_image, edge_clamp, coords - 1).x;
	grad.x -= read_imageui(uc1_src_image, edge_clamp, coords).x;
	grad.y = read_imageui(uc1_src_image, edge_clamp, coords + (int2)(0,-1)).x;
	grad.y -= read_imageui(uc1_src_image, edge_clamp, coords + (int2)(-1,0)).x;

	// compute the square of the gradient magnitude, avoids sqrt and still has same ordering
	int2 grad2 = grad*grad;
	uint mag;
	mag = grad2.x + grad2.y;
	
	// if the strength of the gradient doesn't meet the minimum threshold, no further processing needed
	if (mag < THRESH)
		return;

	mag = native_sqrt((float)mag) * M_SQRT1_2_F;	// this is done to squeeze range into a uchar range

	float f_ang;
	if(grad.x)	// this is a fix so that we can still use the -cl-fast-relaxed-math compiler arg without divide by zero issues
		f_ang = atan2pi((float)grad.y, grad.x);
	else
		f_ang = (grad.y < 0) ? -0.5f : 0.5f;
	
	uchar ang = (char)floor(f_ang * 128) + 32;		// convert to fixed precision and correct for 45 deg offset

	write_imageui(uc2_grad_image, coords, (uint4)(ang, mag, 0, -1));
}
