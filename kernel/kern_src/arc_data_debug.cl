// draws a colorized line from arc start point to endpoint and a black line representing the start tangent and a white line 
// representing the end tangent
#include "arc_data.cl_h"
#include "colorizer.cl_h"
#include "bresenham_line.cl_h"

kernel void arc_data_debug(
	read_only image2d_t ii2_arc_data,
	read_only image2d_t is2_arc_coords,
	write_only image2d_t uc4_out)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	int2 coords = read_imagei(is2_arc_coords, indices).lo;
	if(all(coords == 0))
		return;
	
	ArcData data = ((RW_ArcData)read_imagei(ii2_arc_data, coords).lo).ad;
	uint4 color = (uint4)(scatter_colorize(coords.x ^ (coords.x * coords.y)), -1);

	int2 endpoint = convert_int2(data.endpoint);
	draw_line(coords, coords - 16*convert_int2(data.tangents.lo), (uint4)(0,0,0,-1), uc4_out);
	draw_line(endpoint, endpoint + 16*convert_int2(data.tangents.hi), -1, uc4_out);
	draw_line(coords, endpoint, color, uc4_out);
}