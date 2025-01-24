#include "bresenham_line.cl"
#include "colorizer.cl"

kernel void foci_debug(
	read_only image2d_t ic2_line_data,
	read_only image2d_t us1_seg_in_arc,
	read_only image2d_t ff4_ellipse_foci,
	write_only image2d_t uc4_out_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uint seg_cnt = read_imageui(us1_seg_in_arc, coords).x;

	//arbitrary color to distinguish nearby arcs from each other
	uint4 color = (uint4)(scatter_colorize(coords.x * coords.y), 1000);
	int4 foci;
	if(seg_cnt < 4)
		color /= 3;
	else
	{
		//draw lines between the start of the arc and the associated foci
		foci = convert_int4_sat_rte(read_imagef(ff4_ellipse_foci, coords));
		draw_line(coords, foci.lo, color + 64, uc4_out_image);
		draw_line(coords, foci.hi, color + 64, uc4_out_image);
	}
	//draw lines for all the segments associated with the arc
	for(int i = 0; i < seg_cnt; ++i)
	{
		int2 prev_coords = coords;
		coords += read_imagei(ic2_line_data, coords).lo;
		draw_line(coords, prev_coords, color, uc4_out_image);
		write_imageui(uc4_out_image, prev_coords, -1);
	}

	if(seg_cnt < 4)
		return;
	color.w = 128;
	//draw lines between the end of the arc and the associated foci
	draw_line(coords, foci.lo, color, uc4_out_image);
	draw_line(coords, foci.hi, color, uc4_out_image);

}