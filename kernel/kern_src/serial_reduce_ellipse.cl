kernel void serial_reduce_ellipse(
	read_only image2d_t ii4_group_list,
	write_only image2d_t ii4_reduced_list)
{
	const int bound = get_global_size(0);
	for(int2 i = 0; i.y <= 1; ++i.y)
	{
		int write_pos = 0;

		for(i.x = 0; i.x < bound; ++i.x)
		{
			int4 data = read_imagei(ii4_group_list, i);
			// if the index is not populated, skip it
			if((uint)data.x)
				continue;
			// else proceed to write it to the condensed position
			write_imagei(ii4_reduced_list, (int2)(write_pos, i.y), data);
			// if the top-most bit of data wasn't set, the set continued to the 2nd int4 so read and write that too
			if(data.w >= 0)
			{
				data = read_imagei(ii4_group_list, i | (int2)(0, 2));
				write_imagei(ii4_reduced_list, (int2)(write_pos, 2 | i.y), data);
			}
			++write_pos;
		}
		// terminate with a 0
		write_imagei(ii4_reduced_list, (int2)(write_pos, i.y), 0);
	}
}