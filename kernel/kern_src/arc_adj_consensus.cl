//TODO: sanity check all loop logic here because I've probably some tiny mistake it probably won't be easily noticeable from the final output

// this kernel goes through the arc adjacency links and evaluates which sets of links are mutual and likely to all belong to a
// shared, well-defined, real ellipse in the scene, marks their usage and calculates the ellipse coefficients for later clustering
#include "arc_data.cl_h"
#include "conic_solve.cl_h"
#include "cast_helpers.cl_h"

//NOTE: tuneable minimum coverage in percent required in order for a solution to be written out
#define MIN_COVERAGE_PERCENT	15
// percent scaled to match coverage value scaling
#define MIN_COVERAGE_THRESH		(MIN_COVERAGE_PERCENT * M_1_PI_F / 100)

//TODO: this LUT could be folded in half because the later half is the bitwise inverse of the first half in reverse order
// which could improve cache hit ratio
constant const uchar processed4[70] = {
	0x0F, 0x17, 0x27, 0x47, 0x87, 0x1B, 0x2B, 0x4B, 0x8B, 0x33, 0x53, 0x93, 0x63, 0xA3, 0xC3, 
	0x1D, 0x2D, 0x4D, 0x8D, 0x35, 0x55, 0x95, 0x65, 0xA5, 0xC5, 
	0x39, 0x59, 0x99, 0x69, 0xA9, 0xC9, 
	0x71, 0xB1, 0xD1, 
	0xE1, 
	0x1E, 0x2E, 0x4E, 0x8E, 0x36, 0x56, 0x96, 0x66, 0xA6, 0xC6, 
	0x3A, 0x5A, 0x9A, 0x6A, 0xAA, 0xCA, 
	0x72, 0xB2, 0xD2, 
	0xE2, 
	0x3C, 0x5C, 0x9C, 0x6C, 0xAC, 0xCC, 
	0x74, 0xB4, 0xD4, 
	0xE4, 
	0x78, 0xB8, 0xD8, 
	0xE8, 
	0xF0
};

// computes coverage (and if it's a closed region) for a given clique of arcs and, if it's better than the existing best values
void setBestCliqueIfBetter(int4 const arc_tangents[9], float16 const arc_coeffs[9], float best_elli_gen[5], float* best_coverage, uchar* best_clique, uchar clique_set)
{
//TODO: actually implement closed region check, currently doesn't access the tangents arg and just assumes it's true which is problematic
// due to some solutions, especially for arcs that aren't part of a real ellipse, being numerically unstable and possibly false positives
	float16 elli_coeffs = arc_coeffs[8];

	// add coefficients for all arcs in the clique
	for(int i = 0; i < MAX_CANDIDATES; ++i)
	{
		if(clique_set & (1 << i))
			elli_coeffs += arc_coeffs[i];
	}

	float elli_coverage = elli_coeffs.s8;	//extract perimeter before it gets overwritten
	// solve for the general conic equation coefficients
	float elli_sol[5];
	solveConic((__private float*)&elli_coeffs, elli_sol);

	elli_coverage /= get_ellipse_coverage_divisor(elli_sol);
//	printf("%f ", elli_coverage);
	if(elli_coverage > *best_coverage)
	{
		*best_coverage = elli_coverage;
		*best_clique = clique_set;
		for(int i = 0; i < sizeof(elli_sol); ++i)
			best_elli_gen[i] = elli_sol[i];
	}
}

kernel void arc_adj_consensus(
//	read_only image2d_t ii2_arc_data,
	read_only image2d_t is2_arc_coords,
	read_only image2d_t ff4_pre_solve_coeffs,
	read_only image2d_t ii4_sparse_adj_matrix,
//	write_only image2d_t uc1_adj_consensus,
	write_only image2d_t ff4_sol_coeffs_ABCD,
	write_only image2d_t ff1_sol_coeffs_E)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));

	int2 A_coords[2];
	A_coords[0] = read_imagei(is2_arc_coords, indices).lo;

	// only process initialized arc entries, once there is a null entry, all after are also null
	if(all(A_coords[0] == 0))
		return;

	union s8_conv candidates;
	candidates.i = read_imagei(ii4_sparse_adj_matrix, indices);

	if(all(candidates.i == 0))
	{
//TODO: single arc closed region check and write
		return;
	}
/*//TEMPORARILY DISABLED FOR DEBUGGING
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
*/
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

//TODO: !!! add coverage initialization/processing for single arc closed region
	float best_coverage = 0;
	Conic best_elli_sol;
	uchar best_clique = 0;

	// read in the conic pre-solve coefficients for each of the candidates of arc A
	float16 arc_pre_solves[9];
	readPreSolveCoeffs(ff4_pre_solve_coeffs, A_coords[0], &arc_pre_solves[8]);
	for(int i = 0; i < MAX_CANDIDATES; ++i)
	{
		if(candidates.a[i] < 0)
			break;
		int2 B_coords = read_imagei(is2_arc_coords, (int2)(candidates.a[i], indices.y)).lo;
		readPreSolveCoeffs(ff4_pre_solve_coeffs, B_coords, &arc_pre_solves[i]);
	}

	uchar edge_sets[80] = {0};
	// everything in the local graph has an implicit connection to the primary arc so the size 0 clique is just itself ie this is
	// technically a 1 clique but implementation wise it's 0
	
	// singles include a single other node so are technically 2 cliques but implementation wise are 1 and pairs are technically 3
//	uchar* singles = &edge_sets[sizeof(edge_sets)-(28+8)];
	uchar* pairs = &edge_sets[sizeof(edge_sets)-(28)];
	uchar processed;
//	uchar const pairs = sizeof(edge_sets) - 28;

	// 1 clique processing & converting candidate lists to boolean bit vector form
	for(int i = 0; (i < MAX_CANDIDATES) && (candidates.a[i] >= 0); ++i)
	{
		union s8_conv B_candidates = {.i = read_imagei(ii4_sparse_adj_matrix, (int2)(candidates.a[i], indices.y))};
		// the node gets an edge to itself, this makes some logic simpler
		processed = 1 << i;
		edge_sets[i] = processed;
		// set bits corresponding to shared connections to a node
		for(int j = 0; (j < MAX_CANDIDATES) && (candidates.a[j] >= 0); ++j)
		{
			if(any(B_candidates.s == candidates.a[j]))
				edge_sets[i] |= 1 << j;
		}
		// if possibility set matches processed set, this is a maximal clique
		if(edge_sets[i] == processed)
		{
//TODO: !!! coverage processing
			setBestCliqueIfBetter(NULL, arc_pre_solves, best_elli_sol.general, &best_coverage, &best_clique, processed);
		//	edge_sets[i] = 0;	// this isn't strictly neccessary but helps making it clear that there is no point doing further combining
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
				setBestCliqueIfBetter(NULL, arc_pre_solves, best_elli_sol.general, &best_coverage, &best_clique, processed);
			//	pairs[k] = 0;
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
				processed = processed_1 | processed_2 | (1 << (MAX_CANDIDATES + j - j_thresh));	//TODO: this might be faster with a LUT
				if(processed == (pairs[i] & pairs[j]))
				{
//					printf("3\n");
//TODO: !!! coverage processing
					setBestCliqueIfBetter(NULL, arc_pre_solves, best_elli_sol.general, &best_coverage, &best_clique, processed);
				}
			}
		}
	}

	// 4 clique processing
	for(int i = 0, k = 0, i_step = MAX_CANDIDATES-1, i_thresh = i_step; i_step > 2; i_thresh += i_step)
	{
		--i_step;
		for(int j_step = i_step, j0 = i_thresh; i < i_thresh; ++i)
		{
			j0 += j_step;
			--j_step;
			for(int j = j0; j < 28; ++j, ++k)
			{
				edge_sets[k] = pairs[i] & pairs[j];
				//NOTE: maximal check not done here because it's simpler to do it in the 5+ clique processing stage
				//TODO: check that doing this is actually beneficial perf wise, could potentially be beneficial to add and
				// early exit that checks if 5+ clique processing even has a chance of producing a set or if there are no more
				// shared edges but then, the maximal check for 4 would definitely need to be applied here
			}
		}
	}

	// 5+ clique processing, doesn't require storage
	//TODO: this ordering of iteration is horribly inefficient as it results in approximately 26 duplicate checks on average
	// for each real entry but it should work for now just to see if the idea is working and bug free
	for(int i = 0; i < 70; ++i)
	{
		for(int j = i; j < 70; ++j)
		{
			processed = processed4[i] | processed4[j];
			if((edge_sets[i] & edge_sets[j]) == processed)
			{
//				printf("4+\n");
				setBestCliqueIfBetter(NULL, arc_pre_solves, best_elli_sol.general, &best_coverage, &best_clique, processed);
			}
//TODO: !!! coverage processing
		}
	}

printf("%02X %f\n", best_clique, best_coverage);
	// if no decent match whatsoever, skip writing
	if(best_coverage < MIN_COVERAGE_THRESH)
		return;

	//TODO: consensus probably needs to be stored as candidate list instead for ease of access, could overwrite existing candidate list safely
//	write_imageui(uc1_adj_consensus, indices, best_clique);
	write_imagef(ff4_sol_coeffs_ABCD, indices, best_elli_sol.fm.foci);
	write_imagef(ff1_sol_coeffs_E, indices, best_elli_sol.fm.major);
}