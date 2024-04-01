
// since this is being used with the Hough transform, we want it to repeat on the wrap-around of the angle dimension
// ie. the x dimension, this has the side effect of also wrapping on the y dimension but we can use some padding to
// avoid side effects/artifacts from that
//intended mode for use with hough lines is {REL, {0,-2,0}}
const sampler_t repeat = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
//locates local peaks in images

#define THRESHOLD (32)

// currently a really dumb local max checker that only checks if it's higher than it's 8 immediate neighbors
// [0] In	us1_src_image: 1 channel generic input (UINT16)
// [1] Out	us1_dst_image: 1 channel output (UINT16) that has a value (transferred from src) if
//				it's greater than both a predefined threshold and all 8 surrounding pixels
__kernel void peaks(read_only image2d_t us1_src_image, write_only image2d_t us1_dst_image)
{
	// Determine work item coordinate
	int2 coords = (int2)(get_global_id(0), get_global_id(1) + 1);

	short pixel = read_imageui(us1_src_image, coords).x;
	if(pixel < THRESHOLD)
		return;
	
	short8 neighbors;
	neighbors.s0 = read_imageui(us1_src_image, repeat, coords - 1).x;
	neighbors.s1 = read_imageui(us1_src_image, repeat, coords - (int2)(0,1)).x;
	neighbors.s2 = read_imageui(us1_src_image, repeat, coords - (int2)(-1,1)).x;
	neighbors.s3 = read_imageui(us1_src_image, repeat, coords - (int2)(1,0)).x;
	neighbors.s4 = read_imageui(us1_src_image, repeat, coords + (int2)(1,0)).x;
	neighbors.s5 = read_imageui(us1_src_image, repeat, coords + (int2)(-1,1)).x;
	neighbors.s6 = read_imageui(us1_src_image, repeat, coords + (int2)(0,1)).x;
	neighbors.s7 = read_imageui(us1_src_image, repeat, coords + 1).x;

	if(all(neighbors <= pixel))
		write_imageui(us1_dst_image, coords, (uint4)(pixel, 0, 0, -1));
}