kernel void edge_map_copy(
	read_only image2d_t ic1_grad_ang,
	write_only image2d_t uc4_output)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	if(read_imagei(ic1_grad_ang, coords).x)
		write_imageui(uc4_output, coords, (uint4)(128, 128, 128, -1));
}