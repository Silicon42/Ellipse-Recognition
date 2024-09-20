// draws lines from midpoint of selected arc to candidate matches

#include "colorizer.cl"
#include "path_struct_defs.cl"
#include "bresenham_line.cl"

kernel void arc_adj_debug(
	read_only image2d_t ui4_arc_segments,
	read_only image2d_t iS2_start_coords,
	read_only image1d_t us4_lengths,
	write_only image2d_t uc4_out_image)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	uint4 lengths = read_imageui(us4_lengths, 0);
	// only process valid entries
	if(indices.x >= ((uint*)&lengths)[indices.y])
		return;

	int2 coords = read_imagei(iS2_start_coords, indices).lo;
	union arc_rw arc_raw;
	arc_raw.ui4 = read_imageui(ui4_arc_segments, coords);
	struct arc_data* arc = &arc_raw.data;

	uint4 color;
	color.w = -1;
	switch(indices.y)
	{
	default:	//default to full red for straight
		color.xyz = (uint3)(-1, 0, 0);
		draw_line(coords, coords + convert_int2(arc->offset_end), color, uc4_out_image);
		return;
	case 0:		//ccw, colorize as negative index
		color.xyz = scatter_colorize(-indices.x);
		break;
	case 2:		//cw, colorize as positive index
		color.xyz = scatter_colorize(indices.x);
	}
	//draw_line(coo)
}