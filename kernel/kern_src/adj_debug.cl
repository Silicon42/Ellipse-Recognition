#include "colorizer.cl_h"
#include "bresenham_line.cl_h"

kernel void adj_debug(
	read_only image2d_t is2_arc_coords,
	read_only image2d_t ii4_sparse_adj_matrix,
	write_only image2d_t uc4_out)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	int2 A_coords = read_imagei(is2_arc_coords, indices).lo;
	__attribute__((aligned(16))) short candidates[8];
	*(int4*)candidates = read_imagei(ii4_sparse_adj_matrix, indices);
	uint4 color = (uint4)(scatter_colorize(A_coords.x ^ (A_coords.x * A_coords.y)), -1);

	int2 B_coords;
	for(int i = 0; i < 8 && candidates[i] != -1; ++i)
	{
		B_coords = read_imagei(is2_arc_coords, (int2)(candidates[i], indices.y)).lo;
		draw_line(A_coords, B_coords, color, uc4_out);
	}
}