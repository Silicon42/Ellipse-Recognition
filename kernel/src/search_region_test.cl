// Debugging kernel that mimics the processing of a single primary arc in the same way as arc_adj_matrix.cl
// and highlights regions excluded by checks, un-highlighted region is the final search area
#include "path_struct_defs.cl"

#define POKE_COORDS (int2)(209, 323)//(510, 191)//(530, 647)

kernel void search_region_test(read_only image2d_t ui4_arc_data, read_only image2d_t uc4_lost_seg, write_only image2d_t uc4_debug)
{
	const int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uint4 color = read_imageui(uc4_lost_seg, coords);	//base color inherited from the retrace

	int2 A_start = POKE_COORDS;

	union arc_rw arc_A_packed;
	arc_A_packed.ui4 = read_imageui(ui4_arc_data, A_start);
	struct arc_data* arc_A = &(arc_A_packed.data);
	
	float2 A_end = convert_float2(arc_A->offset_end);	// end offset as float2
	float chord_len = fast_length(A_end);				// chord length of arc
	float2 A_start_f = convert_float2(A_start);			// start coords as float2
	A_end += A_start_f;									// end coords as float2
	float2 A_radial_s = A_start_f - arc_A->center;		// radial vector from center to start
	float2 A_radial_e = A_end - arc_A->center;			// radial vector from center to end

	float radius = fast_length(A_radial_e);				// arc radius
	radius = min(radius, chord_len);	// search radius is lesser of arc radius or chord_len

//From the loop
	// check which location to evaluate for adjacency
	float2 B_start_f = convert_float2(coords);		// start of arc B as float2

	float2 A_to_B = B_start_f - A_end;				// vector from end of arc A to start of arc B
	float dist = fast_length(A_to_B);
	// if it's at or above the max search radius away from the end,
	// skip it, it's not likely part of the same ellipse, also prevents it from
	if(dist >= radius)
		color.x += 64;

	// if start of arc B is outside the tangent line at the end of arc A,
	// A_to_B will have a component in the direction of A_radial_e so dot product will be positive,
	// indicating it should be skipped
	if(dot(A_radial_e, A_to_B) > 0.f)
		color.y += 64;

	// if start of arc B doesn't progress in same direction as arc A's handedness,
	// it can't be the next in the chain of arcs, so skip
	if(cross_2d(A_radial_e, A_to_B) * arc_A->ccw_mult > 0.f)
		color.z += 64;

	//printf("%i\n", arc_A->ccw_mult);
	if(all(coords == convert_int2(arc_A->center)))
		color = -1;
	write_imageui(uc4_debug, coords, color);
}