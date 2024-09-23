// splits flat lines into either cw, ccw, or neither/both based on connection to adjacent segments
#include "cast_helpers.cl"
#include "path_struct_defs.cl"

//NOTE: if performance is bad in this kernel it's probably from false cache invalidation
// via writing to the same buffer as you're reading from, however even if this occurs,
// it should be minimal since the image is so sparse
kernel void flat_split(
	read_only image1d_t iS2_start_coords,
	read_only image2d_t ui4_arc_data,
	write_only image2d_t ui4_arcs_fixed)
{
	union l_conv coords;
	short index = get_global_id(0);	// must be scheduled as 1D

	// initialize variables of arcs segment tracing loop for first iteration
	coords.i = read_imagei(iS2_start_coords, index).lo;
	if(!coords.l)	// this does mean a start at (0,0) won't get processed but I don't think that's particularly likely to happen and be critical
		return;

	union arc_rw arc_raw;
	do
	{
		arc_raw.ui4 = read_imageui(ui4_arc_data, coords.i);

	}while(!arc_raw.data.is_short);

}