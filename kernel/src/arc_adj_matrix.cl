//#include "cast_helpers.cl"
#include "arc_data.cl"
//#include "math_helpers.cl"
#define MAX_CANDIDATES 2	// depending on how well this works this might get increased

//TODO: fix this to work with the split arc list, currently hardcoded to work with the cw list only
kernel void arc_adj_matrix(
	read_only image2d_t iS2_start_coords,
	read_only image1d_t us4_lengths,
	read_only image2d_t ui4_arc_data,
	write_only image1d_t us2_sparse_adj_matrix)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	uint4 lengths = read_imageui(us4_lengths, 0);
	// only process valid entries
	if(indices.x >= ((uint*)&lengths)[indices.y])
		return;

	int2 coords = read_imagei(iS2_start_coords, indices).lo;// finish replacing

	const ushort max_index = read_imageui(us4_lengths, 0).z;	//find how many to iterate over
	const ushort index = get_global_id(0);
	//const uchar arc_dir = get_global_id(1);
	if(index > max_index)
		return;
	
	int2 A_start = read_imagei(iS2_start_coords, index).lo;

	struct arc_data arc_A = read_arc_data(ui4_arc_data, A_start);
	float2 A_end = convert_float2(arc_A.offset_end);	// end offset as float2
	float chord_len = fast_length(A_end);				// chord length of arc
	float2 A_start_f = convert_float2(A_start);			// start coords as float2
	A_end += A_start_f;									// end coords as float2
	float2 A_radial_s = A_start_f - arc_A.center;		// radial vector from center to start
	float2 A_radial_e = A_end - arc_A.center;			// radial vector from center to end

	float radius = fast_length(A_radial_e);				// arc radius
	radius = min(radius, chord_len);	// search radius is lesser of arc radius or chord_len
	
	uchar num_candidates = 0;
	ushort candidates[MAX_CANDIDATES] = {0};
	float candidate_dist[MAX_CANDIDATES] = {INFINITY, INFINITY};
	uchar is_worst = 0;

	//TODO: revisit these checks once you understand the Candy's Theorem constraint, should be more efficient
	for(uint i = 0; i < max_index; ++i)
	{
		// check which location to evaluate for adjacency
		int2 B_start = read_imagei(iS2_start_coords, i).lo;
		float2 B_start_f = convert_float2(B_start);		// start of arc B as float2

		float2 A_to_B = B_start_f - A_end;				// vector from end of arc A to start of arc B
		float dist = fast_length(A_to_B);
		// if it's at or above the max search radius away from the end,
		// skip it, it's not likely part of the same ellipse, also prevents it from
		if(dist >= radius)
			continue;


		// if start of arc B is outside the tangent line at the end of arc A,
		// A_to_B will have a component in the direction of A_radial_e so dot product will be positive,
		// indicating it should be skipped
		if(dot(A_radial_e, A_to_B) > 0)
			continue;

		// if start of arc B doesn't progress in same direction as arc A's handedness,
		// it can't be the next in the chain of arcs, so skip
		if(cross_2d(A_radial_e, A_to_B) * arc_A.ccw_mult > 0)
			continue;

		struct arc_data arc_B = read_arc_data(ui4_arc_data, B_start);

		// check handedness match, if you want mixed handedness for a single ellipse registration,
		// you have your work cut out for you since they register endpoints in the opposite direction
		// or just let them register separately and combine them at the clustering stage
		if(arc_B.ccw_mult != arc_A.ccw_mult)
			continue;

		float2 B_radial_s = B_start_f - arc_B.center;	// radial vector from center to start

		// anti-joggle check
		if(dot(B_radial_s, A_to_B) < 0)
			continue;

		// if the start radial of arc B doesn't progress in angle in direction of handedness
		// relative to the end radial of arc A, then skip it
		//FIXME: it's likely that due to slight over-estimates of arc radius that some that should get matched
		// together here won't quite make the cut and it's not as simple as making a small negative threshold.
		// perhaps a fall back solution that traverses a few pixels in the case of shared start and end points?
		if(cross_2d(A_radial_e, B_radial_s) * arc_A.ccw_mult > 0)
			continue;

		// could add a B chord len search region check here for better symmetry but it would be mostly redundant

		float2 B_end = convert_float2(arc_B.offset_end);	// end offset as float2
		B_end += B_start_f;									// end coords as float2
		float2 B_to_A = A_start_f - B_end;					// vector from end of arc B to start of arc A
		float2 B_radial_e = B_end - arc_B.center;			// radial vector from center to end

		// if start of arc A is outside the tangent line at the end of arc B,
		// B_to_A will have a component in the direction of B_radial_e so dot product will be positive,
		// indicating it should be skipped
		if(dot(B_radial_e, B_to_A) > 0)
			continue;

		// anti-joggle check
		if(dot(A_radial_s, B_to_A) < 0)
			continue;

		// all checks passed, save candidate
		candidates[is_worst] = i;	// replace worst candidate
		candidate_dist[is_worst] = dist;
		is_worst = candidate_dist[0] > candidate_dist[1];	// update worst candidate
		++num_candidates;	// update number of candidates found so far
		if(num_candidates >= MAX_CANDIDATES)
			radius = candidate_dist[is_worst];	// update search range
	}

	write_imageui(us2_sparse_adj_matrix, index, (uint4)(candidates[!is_worst], candidates[is_worst], 0, -1));
}
