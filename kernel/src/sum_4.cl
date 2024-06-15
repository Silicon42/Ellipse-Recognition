// sums 4 pixels, such as corner pixels to face centric conversion for robertsX

__kernel void sum_4(read_only image2d_t iS2_src_image, write_only image2d_t iS4_dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	int2 grad = read_imagei(iS2_src_image, coords).lo;
	grad += read_imagei(iS2_src_image, coords + (int2)(1,0)).lo;
	grad += read_imagei(iS2_src_image, coords + (int2)(0,1)).lo;
	grad += read_imagei(iS2_src_image, coords + 1).lo;

	short ang, mag;
	ang = convert_short_sat_rte(atan2pi((float)grad.y, grad.x) * (1<<15));
	//TODO: add a fix so that we can still use the -cl-fast-relaxed-math compiler arg
	mag = convert_short_rte(fast_length(convert_float2(grad)));

	write_imagei(iS4_dst_image, coords, (int4)(grad, ang, mag));
}