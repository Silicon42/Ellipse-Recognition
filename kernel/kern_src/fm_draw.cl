/*
draws conics and lines to foci for arcs that have 5 or more points associated with them
this kernel is super inefficient in how it draws conics but it's for debugging
purposes only
*/
#include "conic_solve.cl_h"
#include "arc_data.cl_h"
//#include "cast_helpers.cl_h"
#include "colorizer.cl_h"
#include "bresenham_line.cl_h"

kernel void fm_draw(
//	read_only image2d_t ii2_arc_data,
	read_only image2d_t is1_dir_cnt,
	read_only image2d_t ff4_foci,
	read_only image2d_t ff1_major,
	write_only image2d_t uc4_out)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	//ArcData data = ((RW_ArcData)read_imagei(ii2_arc_data, coords).lo).ad;
	short dir_cnt = read_imagei(is1_dir_cnt, coords).x;
	if((dir_cnt & SEG_CNT_MASK) < 4)	// only draw conics that have at least 4 segments to them
		return;

	FociMajor conic = {
		.foci = read_imagef(ff4_foci, coords),
		.major = read_imagef(ff1_major, coords).x
	};

	uint3 color = scatter_colorize(coords.x ^ (coords.x * coords.y));

	int2 bounds = get_image_dim(uc4_out);
	for(int j = 0; j < bounds.y; ++j)
	{
		for(int i = 0; i < bounds.x; ++i)
		{
			float dist = get_conic_deviation(&conic, (float2)(i, j));
			if(!isfinite(dist) || dist > M_SQRT2_F/2)	//actual safety margin is probably M_SQRT2_F
				continue;
			write_imageui(uc4_out, (int2)(i, j), (uint4)(color, -1));
		}
	}

//	int2 end_coords = convert_int2(data.endpoint);
	if(!all(isfinite(conic.foci)))
		return;
	int4 foci = convert_int4_sat_rte(conic.foci);
//	draw_line(end_coords, foci.lo, (uint4)(color/2, 128), uc4_out);
//	draw_line(end_coords, foci.hi, (uint4)(color/2, 128), uc4_out);
	draw_line(coords, foci.lo, (uint4)(color + 64, 128), uc4_out);
	draw_line(coords, foci.hi, (uint4)(color + 64, 128), uc4_out);
}