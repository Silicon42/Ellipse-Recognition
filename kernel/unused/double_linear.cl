// This kernel doubles image resolution using the built-in linear interpolation,

__kernel void double_linear(read_only image2d_t uc1_src_image, write_only image2d_t uc1_dst_image)
{
	const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	write_imageui(uc1_dst_image, coords, read_imageui(uc1_src_image, samp, convert_float2(coords)/2));
}