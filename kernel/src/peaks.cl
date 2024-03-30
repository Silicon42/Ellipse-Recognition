
// since this is being used with the Hough transform, we want it to repeat on the wrap-around of the angle dimension
// ie. the x dimension, this has the side effect of also wrapping on the y dimension but we can use some padding to
// avoid side effects/artifacts from that
//intended mode for use with hough lines is {REL, {0,-2,0}}
const sampler_t repeat = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
//locates local peaks in images

#define THRESHOLD (1<<5)

// currently a really dumb local max checker that only checks if it's higher than it's 8 immediate neighbors
// [0] In	ff1_src_image: 1 channel generic input (UFLOAT)
// [1] OUT	ff1_dst_image: 1 channel output (UFLOAT) that has a value (transferred from src) if
//				it's greater than both a predefined threshold and all 8 surrounding pixels
__kernel void peaks(read_only image2d_t ff1_src_image, write_only image2d_t ff1_dst_image)
{
	// Determine work item coordinate
	int2 coords = (int2)(get_global_id(0), get_global_id(1) + 1);

	float pixel = read_imagef(ff1_src_image, coords).x;
	if(pixel > THRESHOLD)
	{
		float8 neighbors;
		neighbors.s0 = read_imagef(ff1_src_image, repeat, coords - 1).x;
		neighbors.s1 = read_imagef(ff1_src_image, repeat, coords - (int2)(0,1)).x;
		neighbors.s2 = read_imagef(ff1_src_image, repeat, coords - (int2)(-1,1)).x;
		neighbors.s3 = read_imagef(ff1_src_image, repeat, coords - (int2)(1,0)).x;
		neighbors.s4 = read_imagef(ff1_src_image, repeat, coords + (int2)(1,0)).x;
		neighbors.s5 = read_imagef(ff1_src_image, repeat, coords + (int2)(-1,1)).x;
		neighbors.s6 = read_imagef(ff1_src_image, repeat, coords + (int2)(0,1)).x;
		neighbors.s7 = read_imagef(ff1_src_image, repeat, coords + 1).x;

		if(all(neighbors < pixel))
			write_imagef(ff1_dst_image, coords, (float4)(pixel, 0, 0, -1));
	}
}