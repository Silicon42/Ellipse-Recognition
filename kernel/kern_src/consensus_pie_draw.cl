/*
draws lines from centers to associated points for arcs that have 5+ points
associated with them and the segments that connect them
*/
#include "cast_helpers.cl_h"
#include "conic_solve.cl_h"
#include "arc_data.cl_h"
#include "colorizer.cl_h"
#include "draw_pie.cl_h"

kernel void consensus_pie_draw(
	read_only image2d_t uc1_dir_cnt,
	read_only image2d_t is2_arc_coords,
	read_only image2d_t ii4_sparse_adj_matrix,
	read_only image2d_t uc1_adj_consensus,
	read_only image2d_t ff4_foci,
	read_only image2d_t ic2_line_data,
	write_only image2d_t uc4_out)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	float4 foci = read_imagef(ff4_foci, indices);
	if(all(foci == 0))
		return;
	
	int2 center = convert_int2_rte((foci.hi+foci.lo)/2);
	int2 coords = read_imagei(is2_arc_coords, indices).lo;
	if(all(coords == 0))
		return;

	union s8_conv adj_list = {.i = read_imagei(ii4_sparse_adj_matrix, indices)};
	uchar consensus = read_imageui(uc1_adj_consensus, indices).x;
	uint seg_cnt = (read_imageui(uc1_dir_cnt, coords).x & SEG_CNT_MASK) + SEG_CNT_BIAS - 1;
	uint3 color = scatter_colorize(coords.x ^ (coords.x * coords.y));

	draw_pie(ic2_line_data, uc4_out, coords, center, color, seg_cnt);
	for(int i = 0; consensus; ++i)
	{
		// if this adjacency index was part of the clique set
		if(consensus & 1)
		{
			indices.x = adj_list.a[i];
			coords = read_imagei(is2_arc_coords, indices).lo;
			seg_cnt = (read_imageui(uc1_dir_cnt, coords).x & SEG_CNT_MASK) + SEG_CNT_BIAS - 1;
			draw_pie(ic2_line_data, uc4_out, coords, center, color, seg_cnt);
		}
		consensus >>= 1;
	}
}