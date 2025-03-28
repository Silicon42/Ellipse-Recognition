/*
draws ellipses and lines to foci for arcs that have 5 or more points associated with them
this kernel is super inefficient in how it draws ellipses but it's for debugging
purposes only
*/
#include "conic_solve.cl_h"
#include "cast_helpers.cl_h"
#include "colorizer.cl_h"
#include "bresenham_line.cl_h"

kernel void conic_solve_debug(
	read_only image2d_t ff4_pseudo_coeffs,
	write_only image2d_t uc4_out)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	union f16_conv coeffs;
	coeffs.v = 0;
	coeffs.v.hi.hi = read_imagef(ff4_pseudo_coeffs, (int2)(coords.x*4+3, coords.y));
	if(coeffs.v.sf < 5)	// skip drawing if less than 5 points involved
		return;
	coeffs.v.hi.lo = read_imagef(ff4_pseudo_coeffs, (int2)(coords.x*4+2, coords.y));
	coeffs.v.lo.hi = read_imagef(ff4_pseudo_coeffs, (int2)(coords.x*4+1, coords.y));
	coeffs.v.lo.lo = read_imagef(ff4_pseudo_coeffs, (int2)(coords.x*4,   coords.y));

	Ellipse M;
//	printf("%v16f\n", coeffs.v);
	solveConic(coeffs.a, M.general);

	uint4 color = (uint4)(scatter_colorize(coords.x ^ coords.y), -1);
/*
if(any(coords != 0))
	return;
M.general[0] = 3600/155200.f;
M.general[1] = 3680/155200.f;
M.general[2] = -33/155200.f;
M.general[3] = -40/155200.f;
M.general[4] = 24/155200.f;
*/

	convertGeneralConicToFociDistEllipse(&M);
//	printf("%v4f %f,	", M.foci_dist.foci, M.foci_dist.dist);
	draw_line(coords, convert_int2_rte(M.foci_dist.foci.lo), color + 64, uc4_out);
	draw_line(coords, convert_int2_rte(M.foci_dist.foci.hi), color + 64, uc4_out);

	int2 bounds = get_image_dim(uc4_out);
	for(int j = 0; j < bounds.y; ++j)
	{
		for(int i = 0; i < bounds.x; ++i)
		{
			float dist = get_ellipse_deviation(&M.foci_dist, (float2)(i, j));
//			if(i==120 && j==40)
//				printf("%f	", dist);
			if(!isfinite(dist) || dist > M_SQRT2/2)
				continue;
			write_imageui(uc4_out, (int2)(i, j), color);
		}
	}

}