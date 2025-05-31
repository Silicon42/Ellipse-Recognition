// this kernel goes through the arc adjacency links and evaluates which sets of links are mutual and removes ones that aren't
//TODO: it might be better performance wise to wrap this into arc_adj_consensus() for read reduction reasons
#include "conic_solve.cl_h"
#include "cast_helpers.cl_h"

kernel void arc_adj_simplify(
//NOTE: reading and writing to the same buffer like this isn't quite kosher but should be fine so long as the write actually happens
// and a read can't read a partially overwritten value, otherwise you might get duplicate or lost links depending on write order
// even if this does happen, it shouldn't cause any critical issues but there might be inconsistent behavior
	read_only image2d_t ii4_sparse_adj_matrix_r,
	write_only image2d_t ii4_sparse_adj_matrix_w)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));

	union s8_conv A_candidates, A_candidates_out = {.i = -1}, B_candidates;
	A_candidates.i = read_imagei(ii4_sparse_adj_matrix_r, indices);

	for(int i = 0, j = 0; (i < MAX_CANDIDATES) && (A_candidates.a[i] != -1); ++i)
	{
		B_candidates.i = read_imagei(ii4_sparse_adj_matrix_r, (int2)(A_candidates.a[i], indices.y));
		// if B has no matching link to A, remove it from A's list
		if(all(B_candidates.s != (short)indices.x))
			continue;
		
		A_candidates_out.a[j] = A_candidates.a[i];
		++j;
	}

	write_imagei(ii4_sparse_adj_matrix_w, indices, A_candidates_out.i);
}