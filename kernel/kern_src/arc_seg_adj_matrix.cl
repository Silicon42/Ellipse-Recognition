/*
gets up to 8 candidate matches of if arc segment A could be in the same ellipse
as a given arc segment B and if it can, adds the candidate to A's list of up to 8
if there is space.
Only matches arcs of the same turning direction, expects cw arcs in is2_arc_coords
y=0 and ccw arcs in y=1.
*/

//#include "cast_helpers.cl_h"
#include "math_helpers.cl_h"
#include "arc_data.cl_h"
#include "conic_solve.cl_h"
#include "cast_helpers.cl_h"

// state representation of the arc retraction state machine that runs when a pair
// of arc candidates have ends that are too close to be run through the Candy's 
// theorem constraint as is
// bit 0 represents B having both sides retracted
// bits 1 and 2 represent the A retraction degree,
//	0: no retraction,
//	1: tangent retraction,
//	2: 1/4 retraction
//	3: special entry/exit state
enum distState{
	A0B0 = 0b000,
	A0B1 = 0b001,
	A1B0 = 0b010,
	A1B1 = 0b011,
	A2B0 = 0b100,
	A2B1 = 0b101,
	START= 0b110,
	EXIT = 0b111,
};

bool isPointOutOfRegion(int4 tangents, int4 displacements)
{
	tangents *= displacements.yxwz;
	tangents.even -= tangents.odd;
	return any(tangents.even < 0);
}

// Given a test point, the central crossing point, the pre-computed shared portion of the calculation of the Candy's theorem
// that applies to all points, and the foci and major axis length of the suspected arc see if the correspoing point falls 
// within a small margin of error of the edge of the conic
inline bool doesFailCandysCheck(float2 testpoint, float2 const central, float2 const shared_calc, FociMajor const * const B_foci_major)
{
	// express test point coords relative to central point
	testpoint -= central;
	// find corresponding point to test point as according to Candy's Theorem relative to the central point and then add back the central point's offset
	testpoint /= cross_2d_f(testpoint, shared_calc) - 1;
	testpoint += central;

	// check that the predicted point is a close match to arc B's predicted foci and major axis length
	return get_conic_deviation(B_foci_major, testpoint) > M_SQRT2_F;
}

// Fills the test points array with floating point coordinates corresponding to the line segment endpoints of the segments
// that constitute the arc closest to the approximately 5/8, 1/4, 1/2, 3/4, and 3/8 through the arc using the measure of 
// the segment count for the arc, this ensures that chosen points reflect true points on the curve as accurately as possible
// as opposed to interpolating a line segment
// They are ordered this way such that by default the 3 test points are the middle 3 and if a candidate arc is too close to 
// one side, the start point of which 3 to test in the array may simply be shifted +1 or -1 to accomodate
//NOTE: this expects arc segment counts to fit fully in SEG_CNT_MASK (16383 at time of writing) to give accurate results
//NOTE: this traverses the ic2_line_data list for the given arc, so it can be slow for long arcs. I don't currently have a better solution.
inline void getCandysTestPoints(read_only image2d_t ic2_line_data, int seg_cnt, int2 coords, float2 test_points[5])
{
	int test_indices[5];
	test_indices[0] = seg_cnt/4;
	test_indices[1] = (seg_cnt*3)/8;
	test_indices[2] = seg_cnt/2;
	test_indices[3] = seg_cnt - test_indices[1];
	test_indices[4] = seg_cnt - test_indices[0];

	char order[] = {1,4,2,0,3};
	for(int i = 0, cnt = 1; i < 5; )
	{
		if(cnt >= test_indices[i])
		{
			test_points[order[i]] = convert_float2(coords);
			++i;
			continue;
		}

		coords += read_imagei(ic2_line_data, coords).lo;
		++cnt;
	}
}

//TODO: *1 See if a KD tree would help here or if that's too much overhead for the small n

kernel void arc_seg_adj_matrix(
	read_only image2d_t ic2_line_data,
	read_only image2d_t ii2_arc_data,
	read_only image2d_t is1_dir_cnt,
	read_only image2d_t is2_arc_coords,
	read_only image2d_t ff4_ellipse_foci,
	read_only image2d_t ff1_ellipse_major,
	write_only image2d_t ii4_sparse_adj_matrix)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	int2 A_coords[2];
	A_coords[0] = read_imagei(is2_arc_coords, indices).lo;

	// only process initialized arc entries, once there is a null entry, all after are also null
	if(all(A_coords[0] == 0))
		return;

	ArcData A_data = ((RW_ArcData)read_imagei(ii2_arc_data, A_coords[0]).lo).ad;
	A_coords[1] = convert_int2(A_data.endpoint);
	int2 A_end_offset = A_coords[1] - A_coords[0];
	int4 A_tangents = convert_int4(A_data.tangents);

	//TODO: evaluate if using just 2 test points and requiring they both pass is sufficient instead of allowing for 3 with potentially 1 failure
	// if this is the case, the test points could be stored as integers, letting some of the later calculations be integer ops in the absence of an FPU
	// which would reduce the effect of interpolation induced error on low seg_cnt Candy's theorem calcs, additionally you could reduce the array to 4 potential points
	//TODO: this might need to be upped/more intelligently chosen if some close together arcs that should match fail to do so
	float2 test_points[5];
	int seg_cnt = read_imagei(is1_dir_cnt, A_coords[0]).x & SEG_CNT_MASK;
	int2 coords = A_coords[0] + A_tangents.lo;
	if(seg_cnt > 5)
		getCandysTestPoints(ic2_line_data, seg_cnt, coords, test_points);
	else
	{
		test_points[1] = convert_float2(coords);
		test_points[3] = convert_float2(A_coords[1] - A_tangents.hi);
		coords += read_imagei(ic2_line_data, coords).lo;
		test_points[2] = convert_float2(coords);
		test_points[4] = (test_points[1] + test_points[2])/2;
		if(seg_cnt == 4)
			test_points[0] = (test_points[3] + test_points[2])/2;
		else	// seg_cnt == 5
		{
			coords += read_imagei(ic2_line_data, coords).lo;
			test_points[0] = convert_float2(coords);
		}
	}

	// flip vectors for ccw arcs to keep check sense the same
	if(indices.y)
	{
		A_end_offset *= -1;
		A_tangents *= -1;
	}

	int num_candidates = 0;
	union s8_conv candidates = {.i = -1};
	//TODO: *1
	int worst_candidate = 0;
	uint candidate_dist2[MAX_CANDIDATES] = {-1,-1,-1,-1,-1,-1,-1,-1};
	int2 A_avg_coords = A_coords[0] + A_coords[1];

	for(int i = 0; ; ++i)
	{
		// skip matching against itself
		if(indices.x == i)
			continue;
		
		// check which location to evaluate for adjacency
		int2 B_coords[2];
		B_coords[0] = read_imagei(is2_arc_coords, (int2)(i, indices.y)).lo;

		// only process initialized arc entries, once there is a null entry all after are also null
		if(all(B_coords[0] == 0))
			break;

		int4 A_to_B_start;
		A_to_B_start.hi = B_coords[0] - A_coords[1];	// vector from end of arc A to start of arc B
		
		// if start of arc B isn't toward the interior side of arc A,
		// A_end_offset X A_to_B will be negative, indicating it should be skipped
		if(cross_2d_i(A_end_offset, A_to_B_start.hi) < 0)
			continue;

		A_to_B_start.lo = B_coords[0] - A_coords[0];	// vector from start of arc A to start of arc B

		// if the start of B isn't between the tangents of A it should be skipped
		if(isPointOutOfRegion(A_tangents, A_to_B_start))
			continue;

		// since it passed initial tests, read in the tangents and endpoint data for deeper verification
		ArcData B_data = ((RW_ArcData)read_imagei(ii2_arc_data, B_coords[0]).lo).ad;
		B_coords[1] = convert_int2(B_data.endpoint);

		//TODO: *1
		uint dist2 = mag2_2d_i(A_avg_coords - (B_coords[0] + B_coords[1]));
		if(dist2 > candidate_dist2[worst_candidate])
			continue;

		int4 A_to_B_end;
		A_to_B_end.lo = B_coords[1] - A_coords[0];
		// if end of arc B isn't toward the interior side of arc A,
		// A_end_offset X A_to_B will be negative, indicating it should be skipped
		if(cross_2d_i(A_end_offset, A_to_B_end.lo) < 0)
			continue;

		A_to_B_end.hi = B_coords[1] - A_coords[1];

		// if the end of B isn't between the tangents of A it should be skipped
		if(isPointOutOfRegion(A_tangents, A_to_B_end))
			continue;

		int4 B_tangents = convert_int4(B_data.tangents);
		if(indices.y)
			B_tangents *= -1;

		if(isPointOutOfRegion(A_to_B_start, B_tangents.xyxy))
			continue;
		
		if(isPointOutOfRegion(A_to_B_end, B_tangents.zwzw))
			continue;

		// all preliminary region checks passed, do Candy's theorem checks

		// This will be needed later so read it here in hopes that by the time the read latency is up it's actually ready to use
		FociMajor B_foci_major = {.foci = read_imagef(ff4_ellipse_foci, B_coords[0]), .major = read_imagef(ff1_ellipse_major, B_coords[0]).x};
		int2 B_end_offset = B_coords[1] - B_coords[0];

//TODO: re-evaluate the types here once you know more about float vs int performance on different systems, endpoints could be
// represented as ints initially for certain calculations, testpoints must stay floats for low seg_cnt interpolation accuracy reasons
		// check if the distance from A end to B start is within a magnitude of 3 to the distance from A start to B end
		// if it is, then the coords used as the arc endpoints in the Candy's theorem checks need to be changed and the test indices might need to be shifted
		// this is done by getting the squares of the distances and comparing the smaller of them * 8 with the difference
		// if it does not exceed the difference the length is at most 1/3 the length of the longer, in the case where one or both
		// lengths are 0 then this still evaluates as needing a retraction
		float2 A_coords_f[2], B_coords_f[2];	// duplicates for modifying
		A_coords_f[0] = convert_float2(A_coords[0]);
		A_coords_f[1] = convert_float2(A_coords[1]);
		B_coords_f[0] = convert_float2(B_coords[0]);
		B_coords_f[1] = convert_float2(B_coords[1]);
		
		int dist2AB[2];
		dist2AB[0] = mag2_2d_i(A_to_B_end.lo);
		dist2AB[1] = mag2_2d_i(A_to_B_start.hi);
		bool min_sel = 0, min_trend;	// 0 if negative
		char test_index = 1;
		enum distState state = START;
		float const sign_sel[2] = {1,-1};
		//TODO: A_data and B_data shouldn't have to be kept just for this, see if these can be moved for better logistics
		float2 const B_tan[2] = {convert_float2(B_data.tangents.lo), convert_float2(B_data.tangents.hi)};
		float2 const A_tan[2] = {convert_float2(A_data.tangents.lo), convert_float2(A_data.tangents.hi)};
		
		while(state != EXIT)
		{
			dist2AB[min_sel] = mag2_2d_f(B_coords_f[!min_sel] - A_coords_f[min_sel]);
			// + means A start to B end (0) was bigger, - means A end to B start was bigger (1)
			int dist2diffAB = dist2AB[0] - dist2AB[1];
			min_sel = dist2diffAB >= 0;
			// the max length side was >= 3x the length of the min length side
			if(abs(dist2diffAB) < 8 * dist2AB[min_sel])
				break;
			
			switch(state)
			{
			case START:
				state = A0B0;
				B_coords_f[!min_sel] += sign_sel[!min_sel] * B_tan[!min_sel];
				min_trend = min_sel;
				continue;
			case A0B0:
				state = (min_sel == min_trend) ? A1B0 : A0B1;
				if(min_sel == min_trend)
				{
					A_coords_f[min_sel] += sign_sel[min_sel] * A_tan[min_sel];
					test_index += sign_sel[min_sel];
				}
				else
					B_coords_f[!min_sel] += sign_sel[!min_sel] * B_tan[!min_sel];
				continue;
			case A1B0:
				state = (min_sel == min_trend) ? A2B0 : A1B1;
				if(min_sel == min_trend)
					A_coords_f[min_sel] = test_points[1 + 2*min_sel];
				else
					B_coords_f[!min_sel] += sign_sel[!min_sel] * B_tan[!min_sel];
				continue;
			case A2B0:
				state = (min_sel == min_trend) ? EXIT : A2B1;
				if(min_sel != min_trend)
					B_coords_f[!min_sel] += sign_sel[!min_sel] * B_tan[!min_sel];
				continue;
			case A0B1:
				state = A1B1;
				A_coords_f[min_sel] += sign_sel[min_sel] * A_tan[min_sel];
				test_index += sign_sel[min_sel];
				min_trend = min_sel;
				continue;
			case A1B1:
				state = (min_sel == min_trend) ? A2B1 : EXIT;
				if(min_sel == min_trend)
					A_coords_f[min_sel] = test_points[1 + 2*min_sel];
				continue;
			case A2B1:
				state = EXIT;
			case EXIT:
				;
			}
		}

		// Do the parts of the Canny's check calculation that can be shared for each test point

		// get the central point all Canny's checks pass through
		float2 central = intersect_ab_cd(A_coords_f[0], B_coords_f[0], A_coords_f[1], B_coords_f[1]);

		// express A and B end coords relative to start
		float2 A_start_end = A_coords_f[1] - A_coords_f[0];
		float2 B_start_end = B_coords_f[1] - B_coords_f[0];
		// express A and B coords relative to central point
		A_coords_f[0] -= central;
		B_coords_f[0] -= central;
		A_coords_f[1] -= central;
		B_coords_f[1] -= central;
	
		float2 shared = A_start_end / cross_2d_f(A_coords_f[0], A_coords_f[1]) + B_start_end / cross_2d_f(B_coords_f[0], B_coords_f[1]);

		// test if Candy's Theorem constraint passes for at least 2 of the test points
		char fail_cnt = 0;
		// check that the predicted point is a close match to arc B's predicted foci and major axis length
		if(doesFailCandysCheck(test_points[test_index], central, shared, &B_foci_major))
			++fail_cnt;
		
		if(doesFailCandysCheck(test_points[++test_index], central, shared, &B_foci_major))
			++fail_cnt;

	//	printf("%v2f\n", tp_rel);

		switch(fail_cnt)
		{
		default:	// too many failures
	//	printf("?");
			continue;
		case 1:		// could go either way, try 3rd test point
			if(doesFailCandysCheck(test_points[++test_index], central, shared, &B_foci_major))
				continue;
printf("%v2i in Arc_seg_adj_matrix(): B: %i,%i seg_cnt %i\n", A_coords[0], B_coords[0].x, B_coords[0].y, seg_cnt);// debug print to see how often fail of 1 occurs and passes anyways
		case 0:
			;
		}
	//	printf(".");

		// candidate passed all tests, add it to the list overwriting the worst candidate
		//TODO: *1
		candidates.a[worst_candidate] = i;
		candidate_dist2[worst_candidate] = dist2;
		dist2 = 0;
		for(int j = 0; j < MAX_CANDIDATES; ++j)
		{
			if(candidate_dist2[j] > dist2)
			{
				worst_candidate = j;
				dist2 = candidate_dist2[j];
			}
		}
		++num_candidates;
	}
	// arcs with no candidates instead get encoded as 0 to get treated the same as invalid entries
	if(num_candidates == 0)
		return;

	// debug info in case it turns out 8 slots isn't reliably enough in a busy scene
	if(num_candidates > MAX_CANDIDATES)
		printf("%v2i ran out of slots (%i)\n", indices, num_candidates);
	
	write_imagei(ii4_sparse_adj_matrix, indices, candidates.i);
}
