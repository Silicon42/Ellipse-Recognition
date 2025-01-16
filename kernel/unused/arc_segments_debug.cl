#include "bresenham_line.cl"
#include "colorizer.cl"

kernel void arc_segments_debug(
	read_only image2d_t iC2_line_data,
	read_only image2d_t us1_seg_in_arc,
	write_only image2d_t uc4_out_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uint seg_cnt = read_imageui(us1_seg_in_arc, coords).x;

	uint4 color;
	if(seg_cnt < 4)
		color = (uint4)(-1, 0, 0, -1);
	else	//arbitrary color to distinguish nearby arcs from each other
		color = (uint4)(scatter_colorize(coords.x * coords.y), 512);

	//draw lines for all the segments associated with the arc
	for(int i = 0; i < seg_cnt; ++i)
	{
		int2 prev_coords = coords;
		coords += read_imagei(iC2_line_data, coords).lo;
		draw_line(coords, prev_coords, color, uc4_out_image);
		write_imageui(uc4_out_image, prev_coords, -1);
	}
}