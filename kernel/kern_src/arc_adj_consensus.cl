//TODO: sanity check all loop logic here because I've probably some tiny mistake it probably won't be easily noticeable from the final output

// this kernel goes through the arc adjacency links and evaluates which sets of links are mutual and likely to all belong to a
// shared, well-defined, real ellipse in the scene, marks their usage and calculates the ellipse coefficients for later clustering
#include "arc_data.cl_h"
#include "conic_solve.cl_h"
#include "cast_helpers.cl_h"

kernel void arc_adj_consensus(
	read_only image2d_t ic2_line_data,
	read_only image2d_t ii2_arc_data,
	read_only image2d_t is1_dir_cnt,
	read_only image2d_t is2_arc_coords,
	read_only image2d_t ff4_ellipse_foci,
	read_only image2d_t ff1_ellipse_major,
	read_only image2d_t ii4_sparse_adj_matrix,
	write_only image2d_t uc1_adj_consensus,
	write_only image2d_t ff4)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));

	int2 A_coords[2];
	A_coords[0] = read_imagei(is2_arc_coords, indices).lo;

	// only process initialized arc entries, once there is a null entry, all after are also null
	if(all(A_coords[0] == 0))
		return;

	union s8_conv candidates;
	candidates.i = read_imagei(ii4_sparse_adj_matrix, indices);

	ArcData A_data = ((RW_ArcData)read_imagei(ii2_arc_data, indices).lo).ad;
	A_coords[1] = convert_int2(A_data.endpoint);
	int2 A_end_offset = A_coords[1] - A_coords[0];
	int4 A_tangents = convert_int4(A_data.tangents);
	// flip vectors for ccw arcs to keep check sense the same
	if(indices.y)
	{
		A_end_offset *= -1;
		A_tangents *= -1;
	}

	// This is technically a form of maximal clique listing for small node counts where the node count fits within some native type size
	// of bits, which allows for representing graph edges for each node to all other given nodes as a bit vector that can be easily
	// manipulated with bitwise ops. Since in order for arcs to form a clique, they must logically agree with their relative placements
	// to each other, the best coverage of an ellipse solution must be a maximal clique assuming the valid matching region bounded by
	// them is finite. Non-maximal cliques will always have less edge length contributing to coverage for virtually identical ellipse
	// solutions and therefore, worse coverage.[*] This does not give any guarantee about which maximal clique is needed and the scoring
	// does not work similar to an edge or node weighting scheme, so all maximal cliques must be tried with the one providing the best
	// enclosed coverage being selected as the consensus solution. Since the encoding of edges is in bits, the entire working memory
	// can fit in ~80 bytes to calculate the 8 choose 4 = 70 max possible 4 node combinations (technically it's a few bytes less but
	// if I screw up how many exactly, it could very rarely overwrite some of the space that gets used twice before it's finished using
	// it the first time and I don't want to risk that or spend more time for so little gain). Combinations of  <= 4 nodes get made along
	// the way and combinations of > 4 nodes don't need to be stored, only checked for maximal cliques and can be formed out of
	// overlapping combinations of 4

	//NOTE: [*] This does mean that in some cases, using coverage as a metric for matching could erroneously bias towards picking up
	// similar arcs that are nearer, say the inside rim of a mug when trying to match the outside rim but we'll evaluate how much of
	// a problem this is later.

	//NOTE: the primary node is implicit and does not occupy a bit in the bit vector, instead if there is less than MAX_CANDIDATES,
	// then the corresponding leftover singles edge sets (ie edge that it can share) will be empty

//TODO: !!! add coverage initialization/processing


	uchar edge_sets[80] = {0};
	// everything in the local graph has an implicit connection to the primary arc so the size 0 clique is just itself ie this is
	// technically a 1 clique but implementation wise it's 0
	
	// singles include a single other node so are technically 2 cliques but implementation wise are 1 and pairs are technically 3
//	uchar* singles = &edge_sets[sizeof(edge_sets)-(28+8)];
	uchar* pairs = &edge_sets[sizeof(edge_sets)-(28)];
	uchar processed;
//	uchar const pairs = sizeof(edge_sets) - 28;

	// 1 clique processing
	for(int i = 0; (i < MAX_CANDIDATES) && (candidates.a[i] != -1); ++i)
	{
		union s8_conv B_candidates = {.i = read_imagei(ii4_sparse_adj_matrix, (int2)(candidates.a[i], indices.y))};
		// the node gets an edge to itself, this makes some logic simpler
		processed = 1 << i;
		edge_sets[i] = processed;
		// set bits corresponding to shared connections to a node
		for(int j = 0; j < MAX_CANDIDATES; ++j)
		{
			if(any(B_candidates.s == candidates.a[j]))
				edge_sets[i] |= 1 << j;
		}
		// if possibility set matches processed set, this is a maximal clique
		if(edge_sets[i] == processed)
		{
//TODO: !!! coverage processing
			edge_sets[i] = 0;	// this isn't strictly neccessary but helps making it clear that there is no point doing further combining
		}
	}

	// 2 clique processing
	for(int i = 0, k = 0; i < MAX_CANDIDATES; ++i)
	{
		uchar processed_1 = 1 << i;
		for(int j = i+1; j < MAX_CANDIDATES; ++j, ++k)
		{
			processed = processed_1 | (1 << j);
			pairs[k] = edge_sets[i] & edge_sets[j];
			
			if(pairs[k] == processed)
			{
//TODO: !!! coverage processing
				pairs[k] = 0;
			}
		}
	}

	// 3 clique processing, special because it doesn't require storing to edge_sets[]
	// and must be formed with entries with at least one shared node, plus iterated 
	// such that we preferrably avoid duplicate computations
	for(int i = 0, i_step = MAX_CANDIDATES-1, i_thresh = i_step; i_step > 1; i_thresh += i_step)
	{
		uchar processed_1 = 1 << (MAX_CANDIDATES-1 - i_step);
		--i_step;
		for(int j = i_thresh, j_step = i_step, j_thresh = i_thresh + i_step; i < i_thresh; j_thresh += --j_step, ++i)
		{
			uchar processed_2 = 1 << (MAX_CANDIDATES-1 - j_step);
			for(; j < j_thresh; ++j)
			{
				processed = processed_1 | processed_2 | (MAX_CANDIDATES + j - j_thresh);	//TODO: this might be faster with a LUT
				if(processed == (pairs[i] & pairs[j]))
					;
//TODO: !!! coverage processing
			}
		}
	}

	// 4 clique processing
	for(int i = 0, i_step = MAX_CANDIDATES-1, i_thresh = i_step; i_step > 1; i_thresh += i_step)
	{
		uchar processed_1 = 1 << (MAX_CANDIDATES-1 - i_step);
		--i_step;
		for(int j = i_thresh, j_step = i_step, j_thresh = i_thresh + i_step; i < i_thresh; j_thresh += --j_step, ++i)
		{
			uchar processed_2 = 1 << (MAX_CANDIDATES-1 - j_step);
			for(; j < j_thresh; ++j)
			{
				processed = processed_1 | processed_2 | (MAX_CANDIDATES + j - j_thresh);	//TODO: this might be faster with a LUT
				if(processed == (pairs[i] & pairs[j]))
					;
//TODO: !!! coverage processing
			}
		}
	}
	// 5+ clique processing, doesn't require storage
}