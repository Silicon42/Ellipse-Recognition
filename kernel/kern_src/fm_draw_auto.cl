/*
draws conics and lines to foci for arcs that have 5 or more points associated with them
this kernel is super inefficient in how it draws conics but it's for debugging
purposes only
*/
#include "conic_solve.cl_h"
#include "colorizer.cl_h"
#include "bresenham_line.cl_h"

kernel void fm_draw_auto(
	read_only image2d_t ff4_foci,
	read_only image2d_t ff1_major,
	write_only image2d_t uc4_out)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	FociMajor conic = {
		.foci = read_imagef(ff4_foci, coords),
		.major = read_imagef(ff1_major, coords).x
	};
	
	if(all(conic.foci == 0))
		return;
	
	bool isHyperbola = signbit(conic.major);
	float2 approx_center = (conic.foci.lo + conic.foci.hi) / 2;

	uint3 color = scatter_colorize(coords.x ^ (coords.x * coords.y));

	int2 bounds = get_image_dim(uc4_out);
	for(int j = 0; j < bounds.y; ++j)
	{
		for(int i = 0; i < bounds.x; ++i)
		{
			if(isHyperbola && (fast_distance((float2)(i,j), approx_center) > -conic.major))
				continue;
			float dist = get_conic_deviation(&conic, (float2)(i, j));
			if(!isfinite(dist) || dist > M_SQRT1_2_F)	//actual safety margin is probably M_SQRT2_F
				continue;
			write_imageui(uc4_out, (int2)(i, j), (uint4)(color, -1));
		}
	}

	if(!all(isfinite(conic.foci)))
		return;
	printf("%v4f	%f\n", conic.foci, conic.major);
	int4 foci = convert_int4_sat_rte(conic.foci);
	printf("%v2i\n", coords);
	draw_line(foci.lo, foci.hi, (uint4)(color + 64, 128), uc4_out);
}