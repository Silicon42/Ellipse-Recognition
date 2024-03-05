constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// Converts xy vectors to magnitude angle
// [0] In	src_image: 2 channel image of x and y gradient (FLOAT)
// [1] OUT	dst_image: 2 channel image of gradient magnitude (POS FLOAT) and angle (SNORM)
__kernel void mag_ang(read_only image2d_t src_image, write_only image2d_t dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float2 xy = read_imagef(src_image, sampler, coords).lo;
	float2 mag_ang;
	mag_ang.x = length(xy);
	mag_ang.y = atan2(xy.y, xy.x);
	// normalize such that conversion to int types wrap properly
	mag_ang.y *= 2;
	write_imagef(dst_image, coords, (float4)(xy, 0.0, 1.0));
}