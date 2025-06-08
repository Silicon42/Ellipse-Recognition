/*
draws ellipses and lines to foci for arcs that have 5 or more points associated with them
this kernel is super inefficient in how it draws ellipses but it's for debugging
purposes only
*/
#include "conic_solve.cl_h"
#include "arc_data.cl_h"
//#include "cast_helpers.cl_h"
#include "colorizer.cl_h"
#include "bresenham_line.cl_h"

kernel void gen_ellipse_draw(
//	read_only image2d_t is1_dir_cnt,
	read_only image2d_t ff4_abcd,
	read_only image2d_t ff1_e,
	write_only image2d_t uc4_out)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	Ellipse ellipse = {.foci_dist = {
		.foci = read_imagef(ff4_abcd, coords),
		.dist = read_imagef(ff1_e, coords).x
	}};

	FociDist* fd = &ellipse.foci_dist;

	convertGeneralConicToFociDistEllipse(&ellipse);

	if(fd->dist <= 0)
	{
		return;
	}

	uint3 color = scatter_colorize(coords.x ^ (coords.x * coords.y));

	int2 bounds = get_image_dim(uc4_out);
	for(int j = 0; j < bounds.y; ++j)
	{
		for(int i = 0; i < bounds.x; ++i)
		{
			float dist = get_ellipse_deviation(fd, (float2)(i, j));
			if(!isfinite(dist) || dist > M_SQRT2_F/2)	//actual safety margin is probably M_SQRT2
				continue;
			write_imageui(uc4_out, (int2)(i, j), (uint4)(color, -1));
		}
	}

//	int2 end_coords = convert_int2(data.endpoint);
	if(!all(isfinite(fd->foci)))
		return;
	int4 foci = convert_int4_sat_rte(fd->foci);
//	draw_line(end_coords, foci.lo, (uint4)(color/2, 128), uc4_out);
//	draw_line(end_coords, foci.hi, (uint4)(color/2, 128), uc4_out);
	draw_line(coords, foci.lo, (uint4)(color + 64, 128), uc4_out);
	draw_line(coords, foci.hi, (uint4)(color + 64, 128), uc4_out);
}