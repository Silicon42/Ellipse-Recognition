// draws lines from midpoint of selected arc to candidate matches

#include "colorizer.cl"
#include "arc_data.cl"
#include "bresenham_line.cl"

kernel void arc_adj_debug(
	read_only image2d_t ui4_arc_data,
	read_only image2d_t iS2_arc_coords,
	read_only image1d_t us2_lengths,
	read_only image2d_t us2_sparse_adj_matrix,
	write_only image2d_t uc4_out_image)
{
	int index = get_global_id(0);
	char dir = get_global_id(1);
	if(!(index | dir))
		printf("arc_adj_matrix entry\n");

	uint2 lengths = read_imageui(us2_lengths, 0).lo;
	// only process valid entries
	int max_index = ((uint*)&lengths)[dir];
	if(index >= max_index)
		return;

	int2 coords_A, coords_B, mid_A, mid_B;
	struct arc_data arc_A, arc_B;

	coords_A = read_imagei(iS2_arc_coords, (int2)(index, dir)).lo;
	arc_A = read_arc_data(ui4_arc_data, coords_A);
	mid_A = convert_int2(arc_A.offset_mid) + coords_A;

	draw_line(coords_A, coords_A + convert_int2(arc_A.offset_end), (uint4)(-1, 0, 0, -1), uc4_out_image);

	uint2 match_indices = read_imageui(us2_sparse_adj_matrix, (int2)(index, dir)).lo;
	// if no valid candidates, return early
	if(match_indices.x >= max_index)
		return;
	
	int dir_mult = dir ? 1 : -1;
	uint4 color = (uint4)(scatter_colorize(dir_mult * index), -1);

	coords_B = read_imagei(iS2_arc_coords, match_indices.x).lo;
	arc_B = read_arc_data(ui4_arc_data, coords_B);
	mid_B = convert_int2(arc_B.offset_mid) + coords_B;

	draw_line(mid_A, mid_B, color, uc4_out_image);

	// if not valid candidate, return early
	if(match_indices.y >= max_index)
		return;
	
	coords_B = read_imagei(iS2_arc_coords, match_indices.y).lo;
	arc_B = read_arc_data(ui4_arc_data, coords_B);
	mid_B = convert_int2(arc_B.offset_mid) + coords_B;

	draw_line(mid_A, mid_B, color, uc4_out_image);

}