//TODO: the clique finding loops should probably be split into inline functions to improve the readability of the kernel function

// this kernel goes through the arc adjacency links and evaluates which sets of links are mutual and likely to all belong to a
// shared, well-defined, real ellipse in the scene, marks their usage and calculates the ellipse coefficients for later clustering
#include "arc_data.cl_h"
#include "conic_solve.cl_h"
#include "cast_helpers.cl_h"
#include "math_helpers.cl_h"

#if !defined(MAX_CANDIDATES) || (MAX_CANDIDATES != 8)
#error MAX_CANDIDATES determines constants that need to be manually recalculated and the code expects no more 8 candidates, please do so before updating the the candidate count here.

#endif
// PAIR_SETS is MAX_CANDIDATES choose 2 combinations, ie for MAX_CANDIDATES == 8,
// 8! / (2! * (8-2)!) == 28
#define PAIR_SETS	28
//8! / (3! * (8-3)!) == 56
#define TRIPLE_SETS	56
// 8! / (4! * (8-4)!) == 70
#define QUAD_SETS	70

//NOTE: tuneable minimum coverage in percent required in order for a solution to be written out
#define MIN_COVERAGE_PERCENT	48
#define COVERAGE_PERCENT_SCALE	(100 / PERIM_SCALE)
// percent scaled to match coverage value scaling
#define MIN_COVERAGE_THRESH		(MIN_COVERAGE_PERCENT / COVERAGE_PERCENT_SCALE)

//TODO: rework to not need triples again, these don't provide enough benefit for 56 bytes of constant space and 56 bytes of thread local space
constant const uchar processed3[TRIPLE_SETS] = {
	0x07, 0x0B, 0x13, 0x23, 0x43, 0x83,
	0x0D, 0x15, 0x25, 0x45, 0x85,
	0x19, 0x29, 0x49, 0x89,
	0x31, 0x51, 0x91,
	0x61, 0xA1,
	0xC1,
	0x0E, 0x16, 0x26, 0x46, 0x86,
	0x1A, 0x2A, 0x4A, 0x8A,
	0x32, 0x52, 0x92,
	0x62, 0xA2,
	0xC2,
	0x1C, 0x2C, 0x4C, 0x8C,
	0x34, 0x54, 0x94,
	0x64, 0xA4,
	0xC4,
	0x38, 0x58, 0x98,
	0x68, 0xA8,
	0xC8,
	0x70, 0xB0,
	0xD0,
	0xE0
};

//TODO: this LUT could be folded in half because the later half is the bitwise inverse of the first half in reverse order
// which could improve cache hit ratio
constant const uchar processed4[QUAD_SETS] = {
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

constant const uchar quads_idx[MAX_CANDIDATES] = {
	35, 55, 65, 69, 34, 14, 4, 0
};

struct BestEllipse {
	Conic elli;
	float coverage;
	uchar set;
};

// for a given clique of arcs, checks if it's a closed region and if so computes coverage. if it's better than the existing best coverage
// closure overides usage of tangents if corresponding bits in clique_set are set
// set bits in closure mean that the arc in question is a guaranteed 
void setBestCliqueIfBetter(char4 const tangents[9], float16 const arc_coeffs[9], struct BestEllipse* best, uchar closure, uchar clique_set)
{
	bool isClosed = (closure & clique_set) || closure == 0xff;
	// if there is no single closed search region arc in the clique, then it must be checked for a combined closed region
	if(!isClosed)
	{
		char4 region_limits = tangents[8];
		for(int i = 0; i < MAX_CANDIDATES && isClosed; ++i)
		{
			uchar update_bounds = (cross_2d_c(region_limits.lo, tangents[i].hi) <= 0);
			update_bounds |= (cross_2d_c(tangents[i].lo, region_limits.hi) <= 0) << 1;
			switch(update_bounds)
			{
			case 1:	// region_limits.lo needs updating
				region_limits.lo = tangents[i].lo;
				break;
			case 2:	// region_limits.lo needs updating
				region_limits.hi = tangents[i].hi;
				break;
			case 3:
				isClosed = true;
			}
		}
		// if the combined region is not closed, then return
		if(!isClosed)
			return;
	}

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

	elli_coverage /= get_scaled_ellipse_perim(elli_sol);

	if(elli_coverage > best->coverage)
	{
//		printf("%02X\n", clique_set);
		best->coverage = elli_coverage;
		best->set = clique_set;
		for(int i = 0; i < 5; ++i)
			best->elli.general[i] = elli_sol[i];
	}
}

kernel void arc_adj_consensus(
	read_only image2d_t ii2_arc_data,
	read_only image2d_t is2_arc_coords,
	read_only image2d_t ff4_pre_solve_coeffs,
	read_only image2d_t ii4_sparse_adj_matrix,
	write_only image2d_t uc1_adj_consensus,	// currently only for debugging purposes
	write_only image2d_t ff4_sol_coeffs_ACDE,
	write_only image2d_t ff1_sol_coeffs_B)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	int2 A_coords;
	A_coords = read_imagei(is2_arc_coords, indices).lo;
//	if(!all(indices == 0))
//		return;
	// only process initialized arc entries, once there is a null entry, all after are also null
	if(all(A_coords == 0))
		return;

	union s8_conv candidates = {.i = read_imagei(ii4_sparse_adj_matrix, indices)};
	struct BestEllipse best = {0};
	do//while(false)	this allows for breaking out to the write and return section from anywhere in the following block
	{
		char4 tangents[9];
		tangents[8] = (union i_conv){.i = read_imagei(ii2_arc_data, indices).x}.c;

//TODO: see if it's possible to traverse ccw arcs in the reverse direction in some way or save their tangents such that special logic isn't neccessary
		// flip vectors for ccw arcs to keep check sense the same
		if(indices.y)
			tangents[8] *= (char)-1;

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

		// read in the conic pre-solve coefficients for each of the candidates of arc A
		float16 arc_pre_solves[9];
		readPreSolveCoeffs(ff4_pre_solve_coeffs, A_coords, &arc_pre_solves[8]);

		uchar closure = 0;
		// single arc closed region check
		if(cross_2d_c(tangents[8].lo, tangents[8].hi) <= 0)
		{
			closure = -1;
			setBestCliqueIfBetter(tangents, arc_pre_solves, &best, closure, 0);
		}

		if(all(candidates.i == -1))
		{
			if(closure)
				break;	// jump to write-out
			//else
			return;
		}

		for(int i = 0; i < MAX_CANDIDATES; ++i)
		{
			if(candidates.a[i] < 0)
				break;

			int2 B_coords = read_imagei(is2_arc_coords, (int2)(candidates.a[i], indices.y)).lo;
			readPreSolveCoeffs(ff4_pre_solve_coeffs, B_coords, &arc_pre_solves[i]);
			tangents[i] = (union i_conv){.i = read_imagei(ii2_arc_data, (int2)(candidates.a[i], indices.y)).x}.c;
			// flip vectors for ccw arcs to keep check sense the same
			if(indices.y)
				tangents[i] *= (char)-1;
			
			// other arc closed region check
			if(cross_2d_c(tangents[i].lo, tangents[i].hi) <= 0)
				closure |= 1 << i;
		}

		// everything in the local graph has an implicit connection to the primary arc so the size 0 clique is just itself ie this is
		// technically a 1 clique but implementation wise it's 0
		// singles include a single other node so are technically 2 cliques but implementation wise are 1 and pairs are technically 3
		uchar singles[MAX_CANDIDATES] = {0};
		uchar processed;

		// 1 clique processing & converting candidate lists to boolean bit vector form
		for(int i = 0; (i < MAX_CANDIDATES) && (candidates.a[i] >= 0); ++i)
		{
			short8 B_candidates = (union s8_conv){.i = read_imagei(ii4_sparse_adj_matrix, (int2)(candidates.a[i], indices.y))}.s;
			// the node gets an edge to itself, this makes some logic simpler
			processed = 1 << i;
			singles[i] = processed;
			// set bits corresponding to shared connections to a node
			for(int j = 0; (j < MAX_CANDIDATES) && (candidates.a[j] >= 0); ++j)
			{
				if(any(B_candidates == candidates.a[j]))
					singles[i] |= 1 << j;
			}
			// if possibility set matches processed set, this is a maximal clique
			if(processed == singles[i])
			{
				setBestCliqueIfBetter(tangents, arc_pre_solves, &best, closure, processed);
				singles[i] = 0;	// this isn't strictly neccessary but helps making it clear that there is no point doing further combining
			}
		}

//TODO: add early write and exit if singles[] is empty
//		printf("singles %02X %02X %02X %02X %02X %02X %02X %02X\n", singles[0], singles[1], singles[2], singles[3], singles[4], singles[5], singles[6], singles[7]);

		// 2 clique processing
//TODO: These might benefit in terms of speed from applying the singles '&' operations in more pseudo-vector like ways via wider types
		uchar pairs[PAIR_SETS] = {0};
		for(int i = 0, k = 0; i < MAX_CANDIDATES-1; ++i)
		{
			uchar processed_1 = 1 << i;
			uchar single_i = singles[i];
			for(int j = i+1; j < MAX_CANDIDATES; ++j, ++k)
			{
				processed = processed_1 | (1 << j);
				pairs[k] = single_i & singles[j];
				
				if(processed == pairs[k])
				{
					setBestCliqueIfBetter(tangents, arc_pre_solves, &best, closure, processed);
					pairs[k] = 0;
				}
			}
		}

//TODO: add early write and exit if pairs[] is empty
/*		printf("pairs ");
		for(int i = 0; i < PAIR_SETS; ++i)
		{
			printf("%02X ", pairs[i]);
		}
		printf("\n");
*/
		// 3 clique processing
		uchar triples[TRIPLE_SETS] = {0};
		for(int i = 0, k = 0, j0 = 0, jstep = MAX_CANDIDATES; i < MAX_CANDIDATES-2; ++i)
		{
			--jstep;
			j0 += jstep;
			uchar set_i = singles[i];
			for(int j = j0; j < PAIR_SETS; ++j)
			{
				triples[k] = set_i & pairs[j];
				if(processed3[k] == triples[k])
				{
					setBestCliqueIfBetter(tangents, arc_pre_solves, &best, closure, processed3[k]);
					triples[k] = 0;
				}
				++k;
			}
		}

		// 4 clique processing
		uchar quads[QUAD_SETS] = {0};
		for(int i = 0, k = 0, j0 = 0, jstep = PAIR_SETS, jstep2 = MAX_CANDIDATES; i < MAX_CANDIDATES-3; ++i)
		{
			--jstep2;
			jstep -= jstep2;
			j0 += jstep;
			uchar single_i = singles[i];
			for(int j = j0; j < TRIPLE_SETS; ++j)
			{
				quads[k] = single_i & triples[j];
				if(processed4[k] == quads[k])
				{
					setBestCliqueIfBetter(tangents, arc_pre_solves, &best, closure, processed4[k]);
					quads[k] = 0;
				}
				++k;
			}
		}

//TODO: add early write and exit if quads[] is empty
/*		printf("quads ");
		for(int i = 0; i < QUAD_SETS; ++i)
		{
			printf("%02X ", quads[i]);
		}
		printf("\n");
*/
		// 8 clique proccessing, from here on out, clique sets don't require storage
		if(0xFF == (quads[0] & quads[QUAD_SETS-1]))
		{
//TODO: technically this doesn't need the full function since if this one exists it WILL be the best clique but for ease of
// programming, I'm just using the full function here, that being said this should be changed later
			setBestCliqueIfBetter(tangents, arc_pre_solves, &best, closure, 0xFF);
//TODO: this needs to write and return since a clique of 8 will always have the best coverage
		}

		// 7 clique proccessing
		// indexing order is because 5 and 6 cliques also use quads_idx to get step sizes and need it in a different order
		uchar quads_prev = quads[14];
		uchar processed_prev = processed4[14];
		for(int i = 0, idx; i < MAX_CANDIDATES; ++i)
		{
			idx = quads_idx[(i*3) & (MAX_CANDIDATES-1)];
			processed = processed_prev | processed4[idx];
			processed_prev = processed4[idx];
			if((quads_prev & quads[idx]) == processed)
				setBestCliqueIfBetter(tangents, arc_pre_solves, &best, closure, processed);

			quads_prev = quads[idx];
		}

		// 6 clique processing
		for(int i = 0, j0 = 0, jstep = MAX_CANDIDATES; i < MAX_CANDIDATES-5; ++i)
		{
			uchar processed_1 = 1 << i;
			uchar processed_2 = processed_1;
			for(int j = j0; processed_2 & 7; ++j)
			{
				processed_2 <<= 1;
				for(int k = quads_idx[i+1+j-j0]; k < 70; ++k)
				{
					processed = processed_1 | processed_2 | processed4[k];
					if((pairs[j] & quads[k]) == processed)
					{
						setBestCliqueIfBetter(tangents, arc_pre_solves, &best, closure, processed);
					}
				}
			}
			--jstep;
			j0 += jstep;
		}

		// 5 clique processing
		for(int i = 0; i < MAX_CANDIDATES-4; ++i)
		{
			uchar processed_1 = 1 << i;
			for(int j = quads_idx[i]; j < 70; ++j)
			{
				processed = processed_1 | processed4[j];
				if((singles[i] & quads[j]) == processed)
				{
					setBestCliqueIfBetter(tangents, arc_pre_solves, &best, closure, processed);
				}
			}
		}
	}while(false);

printf("%02X %f\n", best.set, best.coverage * COVERAGE_PERCENT_SCALE);

	// if no decent match whatsoever, skip writing
	if(best.coverage < MIN_COVERAGE_THRESH)
		return;

//TODO: consensus probably needs to be stored as candidate list instead for ease of access, could overwrite existing candidate list safely
	write_imageui(uc1_adj_consensus, indices, best.set);
	write_imagef(ff4_sol_coeffs_ACDE, indices, best.elli.fm.foci);
	write_imagef(ff1_sol_coeffs_B, indices, best.elli.fm.major);
}