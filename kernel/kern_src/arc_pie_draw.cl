/*
draws lines from centers to associated points for arcs that have 5+ points
associated with them and the segments that connect them
*/
#include "conic_solve.cl_h"
#include "arc_data.cl_h"
#include "colorizer.cl_h"
#include "draw_pie.cl_h"

kernel void arc_pie_draw(
	read_only image2d_t uc1_dir_cnt,
	read_only image2d_t is2_arc_coords,
	read_only image2d_t ff4_foci,
	read_only image2d_t ic2_line_data,
	write_only image2d_t uc4_out)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	int2 coords = read_imagei(is2_arc_coords, indices).lo;
	if(all(coords == 0))
		return;

	float4 foci = read_imagef(ff4_foci, coords);
	int2 center = convert_int2_rte((foci.lo + foci.hi) / 2);
	uint seg_cnt = (read_imageui(uc1_dir_cnt, coords).x & SEG_CNT_MASK) + SEG_CNT_BIAS - 1;
	uint3 color = scatter_colorize(coords.x ^ (coords.x * coords.y));
//	printf("%i \n", seg_cnt);

	draw_pie(ic2_line_data, uc4_out, coords, center, color, seg_cnt);
}