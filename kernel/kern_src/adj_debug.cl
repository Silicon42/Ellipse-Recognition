#include "colorizer.cl_h"
#include "bresenham_line.cl_h"
#include "cast_helpers.cl_h"

kernel void adj_debug(
	read_only image2d_t is2_arc_coords,
	read_only image2d_t ii4_sparse_adj_matrix,
	write_only image2d_t uc4_out)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	int2 A_coords = read_imagei(is2_arc_coords, indices).lo;
	union s8_conv candidates = {.i = read_imagei(ii4_sparse_adj_matrix, indices)};
	// early exit for unwritten values
	if(all(candidates.i == 0))
		return;
		
	uint4 color = (uint4)(scatter_colorize(A_coords.x ^ (A_coords.x * A_coords.y)), -1);
	printf("%v8i + %i\n", convert_int8(candidates.s), indices.x);

	int2 B_coords;
	for(int i = 0; i < 8 && candidates.a[i] != -1; ++i)
	{
		B_coords = read_imagei(is2_arc_coords, (int2)(candidates.a[i], indices.y)).lo;
		draw_line(A_coords, B_coords, color, uc4_out);
	}
}