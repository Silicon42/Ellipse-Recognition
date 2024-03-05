constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// Converts xy vectors to magnitude angle
// [0] IN	src_image: 2 channel image of x and y gradient (SFLOAT)
// [1] OUT	dst_image: 4 channel image of gradient magnitude (UFLOAT), angle (SNORM), and x and y normalized gradient (SNORM)
__kernel void mag_ang(read_only image2d_t src_image, write_only image2d_t dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float2 xy = read_imagef(src_image, sampler, coords).lo;
	float4 mag_ang;
	mag_ang.x = length(xy);
	// normalize such that conversion to int types wraps properly
	mag_ang.y = 2 * atan2pi(xy.y, xy.x);
	mag_ang.hi = xy / mag_ang.x;
	write_imagef(dst_image, coords, (float4)(xy, 0.0, 1.0));
}