// helper function for drawing adjacent line segments associated with a center point
#include "bresenham_line.cl_h"

void draw_pie(
	read_only image2d_t ic2_line_data,
	write_only image2d_t uc4_out,
	int2 coords,
	int2 center,
	uint3 color,
	uint seg_cnt)
{
	draw_line(center, coords, (uint4)(color, -1), uc4_out);
	for(; seg_cnt > 0; --seg_cnt)
	{
		int2 next_coords = read_imagei(ic2_line_data, coords).lo + coords;
		draw_line(coords, next_coords, (uint4)(color+64, -1), uc4_out);
		draw_line(center, next_coords, (uint4)(color, -1), uc4_out);
		coords = next_coords;
	}
}