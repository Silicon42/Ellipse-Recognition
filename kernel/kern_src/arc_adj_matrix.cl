#include "cast_helpers.cl"
#include "math_helpers.cl"

kernel void arc_adj_matrix(
	read_only image2d_t iC2_line_data,
	read_only image2d_t iS2_line_coords,
	read_only image1d_t us1_length,
	write_only image2d_t us4_sparse_adj_matrix)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	ushort max_index = read_imageui(us1_length, 0).x;

	// only process valid entries
	if(((indices.y << 8) | indices.x) > max_index)
		return;

	int2 A_start = read_imagei(iS2_line_coords, indices).lo;
	int2 A_end_offset = read_imagei(iC2_line_data, A_start).lo;
	int2 A_end = A_start + A_end_offset;
	uint worst_dist2 = mag2_2d_i(A_end_offset);
	
	uchar num_candidates[2] = {0};
	ushort candidates[3] = {-1, -1, -1};
	uint candidate_dist2[3] = {-1, -1, -1};
	uchar worst = 0;
	uchar is_ccw, is_both;
	int2 B_start, A_to_B, B_end_offset;
	uint dist2;

	//TODO: revisit these checks once you understand the Candy's Theorem constraint, should be more efficient
	for(uint i = 0; i <= max_index; ++i)
	{
		// check which location to evaluate for adjacency
		B_start = read_imagei(iS2_line_coords, SPLIT_INDEX(i)).lo;

		A_to_B = B_start - A_end;	// vector from end of segment A to start of segment B
		dist2 = mag2_2d_i(A_to_B);
		// if it's at or above the max search radius away from the end,
		// skip it, it's not likely part of the same ellipse,
		// also prevents it from including itself
		if(dist2 >= worst_dist2)
			continue;

		// if start of segment B isn't forward of the end of segment A,
		// A_to_B will have a component against the direction of A_end_offset
		// so dot product will be negative, indicating it should be skipped
		if(dot_2d_i(A_end_offset, A_to_B) < 0)
			continue;

		B_end_offset = read_imagei(iC2_line_data, B_start).lo;

		// angle between segments A and B must be acute, ie positive dot product
		if(dot_2d_i(A_end_offset, B_end_offset) <= 0)
			continue;
		
		int dir = cross_2d_i(A_end_offset, B_end_offset);
		// anti-joggle check, the turning direction of the segment offsets must
		// match that of the line between them, meaning the product of the 2 must be non-negative
		if(dir * cross_2d_i(A_end_offset, A_to_B) < 0)
			continue;

		// could add a B chord len search region check here for better symmetry but it would be mostly redundant
		
		is_ccw = dir <= 0;
		is_both = dir == 0;
		// all checks passed, save candidate
		//TODO: this might be done better with vector selection
		do	// the indexing is done to avoid branching since any given arc will typically find one or the other but almost never both
		{
			candidates[is_worst[is_ccw]][is_ccw] = i;	// replace worst candidate
			candidate_dist2[is_worst[is_ccw]][is_ccw] = dist2;
			is_worst[is_ccw] = candidate_dist2[0][is_ccw] > candidate_dist2[1][is_ccw];	// update worst candidate
			++num_candidates[is_ccw];	// update number of candidates found so far
			//if(num_candidates[is_ccw] >= MAX_CANDIDATES)
				worst_dist2 = candidate_dist2[is_worst[is_ccw]][is_ccw];	// update search range
		}while(is_both--);
	}
	printf("(%i,%i) %u %u %u %u\n", A_start, candidates[0][0], candidates[1][0], candidates[0][1], candidates[1][1]);
	write_imageui(us4_sparse_adj_matrix, indices, (uint4)(candidates[!is_worst[0]][0], candidates[is_worst[0]][0], candidates[!is_worst[1]][1], candidates[is_worst[1]][1]));
}
