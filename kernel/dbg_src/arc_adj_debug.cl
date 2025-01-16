// draws lines from midpoint of selected arc to candidate matches

#include "colorizer.cl"
#include "bresenham_line.cl"

kernel void arc_adj_debug(
	read_only image2d_t iC2_line_data,
	read_only image2d_t iS2_line_coords,
	read_only image1d_t us1_length,
	read_only image2d_t us4_sparse_adj_matrix,
	write_only image2d_t uc4_out_image)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	ushort max_index = read_imageui(us1_length, 0).x;

	// only process valid entries
	if(((indices.y << 8) | indices.x) > max_index)
		return;

	int2 A_start, A_end_offset, A_end, A_mid, B_start, B_mid;
	A_start = read_imagei(iS2_line_coords, indices).lo;
	A_end_offset = read_imagei(iC2_line_data, A_start).lo;
	A_end = A_start + A_end_offset;
	A_mid = A_start + A_end_offset/2;
	
	uint4 color = (uint4)(scatter_colorize(indices.x | (indices.y << 8)), -1);
	draw_line(A_start, A_end, color/2, uc4_out_image);
	write_imageui(uc4_out_image, A_start, color);
	write_imageui(uc4_out_image, A_end, color);

	union ui4_array match_indices;
	match_indices.ui4 = read_imageui(us4_sparse_adj_matrix, indices);
	// plot matches if they exist
	for(int i = 0; i < 4; ++i)
	{
		if(match_indices.arr[i] > max_index)
			continue;
		
		B_start = read_imagei(iS2_line_coords, SPLIT_INDEX(match_indices.arr[i])).lo;
		B_mid = read_imagei(iC2_line_data, B_start).lo/2 + B_start;
		draw_line(A_mid, B_mid, color, uc4_out_image);
	}
}