kernel void input_copy(
	read_only image2d_t fu1_input,
	write_only image2d_t uc4_output)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uchar pixel = 255 * read_imagef(fu1_input, coords).x;
	write_imageui(uc4_output, coords, (uint4)(pixel, pixel, pixel, -1));
}