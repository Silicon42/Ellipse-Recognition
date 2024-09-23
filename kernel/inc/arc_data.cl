#ifndef ARC_DATA_CL
#define ARC_DATA_CL

#include "offsets_LUT.cl"
#include "math_helpers.cl"

#define IS_END			0x01
#define IS_CW			0x02
#define IS_FLAT			0x04
#define IS_BOTH_HANDED	0x08

struct arc_data{	//TODO: check how to ensure optimal packing
	char len;			// arc length in pixels traversed, min of 1, max of 127
	char flags;			// collection of flags: 0b0000bfce	// no bitfield suppprt in OpenCL :(
	// "e" represents arc segment creation thread ended processing on this arc,
	// "c" represents if the arc is a cw processing arc (gradient outwards), multiplier extracted by mask then -1
	// "f" represents if the arc segment was too flat to reliably tell curve direction,
	// "b" represents if the arc should be counted to both cw and ccw processing due to ambiguity (only for flat arcs)
	char2 offset_aux;	// offset from start to auxiliary point, used for verifying ellipse matches
	char2 offset_end;	// offset from start to end point
	char2 offset_mid;	// offset from start to approximate arc midpoint
	float2 center;		// rough estimate of the center coords of the arc
};

// used only for reading/writing arc_data to an image buffer
union _arc_rw{
	struct arc_data data;
	uint4 ui4;
};

//len, end flag, and coords must be pre-populated, rest get calculated here before writing
void write_arc_data(write_only image2d_t ui4_arc_data, int2 base_coords, int2 prev_coords, struct arc_data arc[2], char curr)
{
	// determine state of flag for flatness which we define as the midpoint of the chord having a small displacement
	// from the approximate halfway point of the arc, in the case of a nearly flat arc the halfway point should be 
	// nearly exactly the arc midpoint and very near to the chord midpoint
	const int2 offset_end = convert_int2(arc[curr].offset_end);
	const int2 offset_mid = convert_int2(arc[curr].offset_mid);
	int2 mid_displacement = 2*offset_mid - offset_end;
	int disp_dist2 = mag2_2d_i(mid_displacement);
	char is_flat = (disp_dist2 <= 2) << 2;
	char is_prev_flat = arc[!curr].flags & IS_FLAT;
	arc[curr].flags |= is_flat;
	char prev_length = arc[!curr].len;
	//TODO: verify following approximations are good enough for calculations where one or the other is a flat arc
	// or replace it with a more accurate calculation
	char2 tangent_approx, prev_tangent_approx;
	tangent_approx = arc[curr].offset_end;
	if(!is_flat)
		tangent_approx -= arc[curr].offset_mid;

	prev_tangent_approx = arc[!curr].offset_end;
	if(!is_prev_flat)
		prev_tangent_approx -= arc[!curr].offset_mid;

	char is_cw_approx = (cross_2d_c(prev_tangent_approx, tangent_approx) > 0) << 1;

	// if the prev segment was flat and it rotated differently from what the approximation says,
	// then it needs to flagged for use in both cw and ccw processing and updated
	if(((arc[!curr].flags ^ is_cw_approx) & (IS_FLAT | IS_CW)) == (IS_FLAT | IS_CW))
	{
		arc[!curr].flags |= IS_BOTH_HANDED;
		write_imageui(ui4_arc_data, prev_coords, ((union _arc_rw)arc[!curr]).ui4);
	}
	
	int2 perp_end = perp_2d_i(offset_end);

	// if mid and end points are prefectly in line, ie it's flat, the scale div will be zero
	if(is_flat)	//prevent chance of divide by zero
	{
		if(arc[!curr].len)	// if it wasn't the first in the processing chain
		{
			arc[curr].flags |= is_cw_approx;
			perp_end *= is_cw_approx - 1;
		}
		else if(arc[curr].flags & IS_END)	// no context available, just straight segment
			arc[curr].flags |= IS_BOTH_HANDED;
		//else default state is assumed ccw

		arc[curr].center = convert_float2(perp_end) * -1099511627776.f;	// 2^40 to make it lose all fine detail at 2^16 scale
	}
	else
	{
		// how much to divide the center direction vector by to get center location
		int scale_div = 2 * cross_2d_i(offset_end, offset_mid);
		// cw arcs have cross product of midpoint to endpoint positive and ccw negative
		// since scale_div already includes a scaled version of the cross product, just use that
		arc[curr].flags |= (scale_div > 0) << 1;

		//solve for center of circle
		int mag2_end, mag2_mid;
		mag2_end = mag2_2d_i(offset_end);
		mag2_mid = mag2_2d_i(offset_mid);

		// center direction relative to start, not to scale yet
		int2 center_dir = mag2_mid * perp_end - mag2_end * perp_2d_i(offset_mid);
		//scale correctly, now it's a relative offset to start
		float2 center = convert_float2(center_dir) / scale_div;

		center = convert_float2(base_coords) + center;
		arc[curr].center = center;
	}

	write_imageui(ui4_arc_data, base_coords, ((union _arc_rw)arc[curr]).ui4);
}

inline struct arc_data read_arc_data(read_only image2d_t ui4_arc_data, int2 coords)
{
	return ((union _arc_rw)read_imageui(ui4_arc_data, coords)).data;
}

#endif//ARC_DATA_CL