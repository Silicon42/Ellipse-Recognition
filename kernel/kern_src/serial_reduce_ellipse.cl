kernel void serial_reduce_ellipse(
	read_only image2d_t ii4_group_list,
	write_only image2d_t ii4_reduced_list)
{
	int bound = get_global_size(0);
	for(int i = 0; i <= 1; ++i)
	{
		for(int j = 0; j < bound; ++j)
		{
			 read_imagei(ii4_group_list, (int2)(i,j));
		}
	}
}