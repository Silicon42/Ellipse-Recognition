/*
stripped down version of arc_seg_adj_matrix() for debugging / logic checking,
doesn't use Candy's theorem constraints
*/

#include "math_helpers.cl_h"
#include "arc_data.cl_h"
#include "conic_solve.cl_h"
#include "cast_helpers.cl_h"
#include "bresenham_line.cl_h"
#include "colorizer.cl_h"

bool isPointOutOfRegion1(int4 tangents, int4 displacements)
{
	tangents *= displacements.yxwz;
	tangents.even -= tangents.odd;
	return any(tangents.even < 0);
}

kernel void arc_seg_adj_stripped(
//	read_only image2d_t ic2_line_data,
	read_only image2d_t ii2_arc_data,
//	read_only image2d_t is1_dir_cnt,
	read_only image2d_t is2_arc_coords,
//	read_only image2d_t ff4_ellipse_foci,
//	read_only image2d_t ff1_ellipse_major,
	write_only image2d_t uc4_out)
{
	int2 B_coords[2];
	B_coords[0] = B_coords[1] = (int2)(get_global_id(0), get_global_id(1));
	int2 A_coords[2];
	int2 indices = (int2)(1,0);
	A_coords[0] = read_imagei(is2_arc_coords, indices).lo;

	// only process initialized arc entries, once there is a null entry, all after are also null
	if(all(A_coords[0] == 0))
		return;

	ArcData A_data = ((RW_ArcData)read_imagei(ii2_arc_data, A_coords[0]).lo).ad;
	A_coords[1] = convert_int2(A_data.endpoint);
	int2 A_end_offset = A_coords[1] - A_coords[0];
	int4 A_tangents = convert_int4(A_data.tangents);
	if(all(B_coords[0] == 0))
	{
		printf("%v2i", A_coords[1]);
		draw_line(A_coords[0], A_coords[1], MAGENTA, uc4_out);		// (chord)
		draw_line(A_coords[0], A_coords[0] - A_tangents.lo*256, YELLOW, uc4_out);	// (start tangent)
		draw_line(A_coords[1], A_coords[1] + A_tangents.hi*256, CYAN, uc4_out);	// (end tangent)
	}
	// flip vectors for ccw arcs to keep check sense the same
	if(indices.y)
	{
		A_end_offset *= -1;
		A_tangents *= -1;
	}

//	for(int i = 0; ; ++i)
	while(((B_coords[0].x ^ B_coords[0].y) & 3) == 0)	//while just so I can break out without changing code too much
	{
		int4 A_to_B_start;
		A_to_B_start.hi = B_coords[0] - A_coords[1];	// vector from end of arc A to start of arc B
		
		// if start of arc B isn't toward the interior side of arc A,
		// A_end_offset X A_to_B will be negative, indicating it should be skipped
		if(cross_2d_i(A_end_offset, A_to_B_start.hi) < 0)
			break;

		A_to_B_start.lo = B_coords[0] - A_coords[0];	// vector from start of arc A to start of arc B

		// if the start of B isn't between the tangents of A it should be skipped
		if(isPointOutOfRegion1(A_tangents, A_to_B_start))
			break;

		// since it passed initial tests, read in the tangents and endpoint data for deeper verification
	//	ArcData B_data = ((RW_ArcData)read_imagei(ii2_arc_data, B_coords[0]).lo).ad;
	//	B_coords[1] = convert_int2(B_data.endpoint);

		int4 A_to_B_end;
		A_to_B_end.lo = B_coords[1] - A_coords[0];
		// if end of arc B isn't toward the interior side of arc A,
		// A_end_offset X A_to_B will be negative, indicating it should be skipped
		if(cross_2d_i(A_end_offset, A_to_B_end.lo) < 0)
			break;

		A_to_B_end.hi = B_coords[1] - A_coords[1];

		// if the end of B isn't between the tangents of A it should be skipped
		if(isPointOutOfRegion1(A_tangents, A_to_B_end))
			break;
/*
		int4 B_tangents = convert_int4(B_data.tangents);
		if(indices.y)
			B_tangents *= -1;

		if(isPointOutOfRegion1(A_to_B_start, B_tangents.xyxy))
			break;
		
		if(isPointOutOfRegion1(A_to_B_end, B_tangents.zwzw))
			break;
*/
		// all preliminary region checks passed, do Candy's theorem checks

		write_imageui(uc4_out, B_coords[0], GRAY);
		break;
	}
//	write_imageui(uc4_out, B_coords[0], -1);
}
