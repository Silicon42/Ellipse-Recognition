const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

//TODO: consider adding "Magic Kernel Sharp" to preserve higher edge resolution
// as detailed here: https://johncostella.com/edgedetect/

// [0] In	src_image: 1 channel greyscale on x component (UNORM)
// [1] Out	dst_image: 2 channel image of x and y gradient (SFLOAT)
__kernel void scharr(read_only image2d_t src_image, write_only image2d_t dst_image)
{
	// itermediate sums for horizontal/vertical scharr shared operations
	float diag1, diag2;
	float2 grad;
	// Determine output coordinate
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	// calculate intermediate sums from raw pixels
	diag1 = read_imagef(src_image, sampler, coords - 1).x;
	diag1 -= read_imagef(src_image, sampler, coords + 1).x;
	diag2 = read_imagef(src_image, sampler, coords - (int2)(1,-1)).x;
	diag2 -= read_imagef(src_image, sampler, coords + (int2)(1,-1)).x;
	grad.x = read_imagef(src_image, sampler, coords - (int2)(1,0)).x;
	grad.x -= read_imagef(src_image, sampler, coords + (int2)(1,0)).x;
	grad.y = read_imagef(src_image, sampler, coords - (int2)(0,1)).x;
	grad.y -= read_imagef(src_image, sampler, coords + (int2)(0,1)).x;

	grad = fma(grad, 3.44680851f, diag1);
	grad += (float2)(diag2,-diag2);
	//re-normalize values
	//grad = fma(grad, 0.091796875f, 0.5f);

	write_imagef(dst_image, coords, (float4)(grad, 0.5, 1.0));

}