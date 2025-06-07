/*
stripped down version of the candy's theorem logic from arc_seg_adj_matrix() that draws the
lines for the intermediate calculations of a pre-set pair of arcs for debugging purposes
*/

//#include "cast_helpers.cl_h"
#include "math_helpers.cl_h"
#include "arc_data.cl_h"
#include "conic_solve.cl_h"
#include "cast_helpers.cl_h"
#include "bresenham_line.cl_h"
#include "colorizer.cl_h"

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

kernel void candys_theorem_example(
	read_only image2d_t ic2_line_data,
	read_only image2d_t ii2_arc_data,
	read_only image2d_t is1_dir_cnt,
	read_only image2d_t is2_arc_coords,
	read_only image2d_t ff4_ellipse_foci,
	read_only image2d_t ff1_ellipse_major,
	write_only image2d_t uc4_out)
{
	int2 indices = (int2)(8,0);
	int B_index = 7;
	int2 A_coords[2];
	A_coords[0] = read_imagei(is2_arc_coords, indices).lo;
	ArcData A_data = ((RW_ArcData)read_imagei(ii2_arc_data, A_coords[0]).lo).ad;
	A_coords[1] = convert_int2(A_data.endpoint);
	int2 A_end_offset = A_coords[1] - A_coords[0];
	int4 A_tangents = convert_int4(A_data.tangents);
	
	int2 B_coords[2];
	B_coords[0] = read_imagei(is2_arc_coords, (int2)(B_index, indices.y)).lo;
	ArcData B_data = ((RW_ArcData)read_imagei(ii2_arc_data, B_coords[0]).lo).ad;
	B_coords[1] = convert_int2(B_data.endpoint);
	int4 B_tangents = convert_int4(B_data.tangents);

	// flip vectors for ccw arcs to keep check sense the same
	if(indices.y)
	{
		A_end_offset *= -1;
		A_tangents *= -1;
		B_tangents *= -1;
	}

	draw_line(A_coords[0], A_coords[1], CYAN, uc4_out);
	draw_line(B_coords[0], B_coords[1], YELLOW, uc4_out);
	draw_line(A_coords[0], B_coords[1], MAGENTA/2, uc4_out);
	draw_line(B_coords[0], A_coords[1], MAGENTA/2, uc4_out);
	draw_line(A_coords[0], A_coords[0]+A_tangents.lo, WHITE, uc4_out);
	draw_line(A_coords[1], A_coords[1]-A_tangents.hi, GRAY, uc4_out);
	draw_line(B_coords[0], B_coords[0]+B_tangents.lo, WHITE, uc4_out);
	draw_line(B_coords[1], B_coords[1]-B_tangents.hi, GRAY, uc4_out);


	int4 A_to_B_start;
	A_to_B_start.hi = B_coords[0] - A_coords[1];	// vector from end of arc A to start of arc B
	int4 A_to_B_end;
	A_to_B_end.lo = B_coords[1] - A_coords[0];

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


//	for(int i = 0; ; ++i)
	{
		// all preliminary region checks passed, do Candy's theorem checks

		// This will be needed later so read it here in hopes that by the time the read latency is up it's actually ready to use
		FociDist B_foci_major = {.foci = read_imagef(ff4_ellipse_foci, B_coords[0]), .dist = read_imagef(ff1_ellipse_major, B_coords[0]).x};
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
		float2 const B_tan[2] = {convert_float2(B_tangents.lo), convert_float2(B_tangents.hi)};
		float2 const A_tan[2] = {convert_float2(A_tangents.lo), convert_float2(A_tangents.hi)};
		
		while(state != EXIT)
		{
			dist2AB[min_sel] = mag2_2d_f(B_coords_f[!min_sel] - A_coords_f[min_sel]);
			// + means A start to B end (0) was bigger, - means A end to B start was bigger (1)
			int dist2diffAB = dist2AB[0] - dist2AB[1];
			min_sel = dist2diffAB >= 0;
			printf("\ndist2AB = {%i, %i},	min_sel = %i	state = ", dist2AB[0], dist2AB[1], min_sel);
			// the max length side was < 3x the length of the min length side, no retraction of endpoints neccessary
			if(abs(dist2diffAB) < 8 * dist2AB[min_sel])
				break;
			
			switch(state)
			{
			case START:
				printf("Start	");
				state = A0B0;
				B_coords_f[!min_sel] += sign_sel[!min_sel] * B_tan[!min_sel];
				min_trend = min_sel;
				continue;
			case A0B0:
				printf("A0B0	");
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
				printf("A1B0	");
				state = (min_sel == min_trend) ? A2B0 : A1B1;
				if(min_sel == min_trend)
					A_coords_f[min_sel] = test_points[1 + 2*min_sel];
				else
					B_coords_f[!min_sel] += sign_sel[!min_sel] * B_tan[!min_sel];
				continue;
			case A2B0:
				printf("A2B0	");
				state = (min_sel == min_trend) ? EXIT : A2B1;
				if(min_sel != min_trend)
					B_coords_f[!min_sel] += sign_sel[!min_sel] * B_tan[!min_sel];
				continue;
			case A0B1:
				printf("A0B1	");
				state = A1B1;
				A_coords_f[min_sel] += sign_sel[min_sel] * A_tan[min_sel];
				test_index += sign_sel[min_sel];
				min_trend = min_sel;
				continue;
			case A1B1:
				printf("A1B1	");
				state = (min_sel == min_trend) ? A2B1 : EXIT;
				if(min_sel == min_trend)
					A_coords_f[min_sel] = test_points[1 + 2*min_sel];
				continue;
			case A2B1:
				printf("A2B1	");
				state = EXIT;
			case EXIT:
				;
			}
		}
		//draw updated dist lines
		draw_line(convert_int2(A_coords_f[0]), convert_int2(B_coords_f[1]), GREEN, uc4_out);
		draw_line(convert_int2(B_coords_f[0]), convert_int2(A_coords_f[1]), GREEN, uc4_out);


		// Do the parts of the Canny's check calculation that can be shared for each test point

		// get the central point all Canny's checks pass through
		float2 central = intersect_ab_cd(A_coords_f[0], B_coords_f[0], A_coords_f[1], B_coords_f[1]);

		// express A and B end coords relative to start
		A_coords_f[1] -= A_coords_f[0];
		B_coords_f[1] -= B_coords_f[0];
		// express A and B start coords relative to central point
		A_coords_f[0] -= central;
		B_coords_f[0] -= central;

		float2 shared = A_coords_f[1] / cross_2d_f(A_coords_f[0], A_coords_f[1]) + B_coords_f[1] / cross_2d_f(B_coords_f[0], B_coords_f[1]);

		// test if Candy's Theorem constraint passes for at least 2 of the test points
		float2 tp_rel;
		char fail_cnt = 0;
		// express test point coords relative to central point
		tp_rel = test_points[test_index] - central;
		// find corresponding point to test point as according to Candy's Theorem relative to the central point and then add back the central point's offset
		tp_rel *= cross_2d_f(tp_rel, shared);
		tp_rel += central;

		// check that the predicted point is a close match to arc B's predicted foci and major axis length
		if(get_ellipse_deviation(&B_foci_major, tp_rel) > M_SQRT2_F)
			++fail_cnt;
		
		++test_index;
		// do the above again for the second test point
		tp_rel = test_points[test_index] - central;
		tp_rel *= cross_2d_f(tp_rel, shared);
		tp_rel += central;
		if(get_ellipse_deviation(&B_foci_major, tp_rel) > M_SQRT2_F)
			++fail_cnt;
		
		printf("%v2f\n", tp_rel);
/*
		switch(fail_cnt)
		{
		default:	// too many failures or invalid (How???)
			continue;
		case 1:	// could go either way, try 3rd test point
			++test_index;
			tp_rel = test_points[test_index] - central;
			tp_rel *= cross_2d_f(tp_rel, shared);
			tp_rel += central;
			if(get_ellipse_deviation(&B_foci_major, tp_rel) > M_SQRT2_F)
			{
				continue;
			}
printf("%v2i in Arc_seg_adj_matrix(): B: %i,%i seg_cnt %i\n", A_coords[0], B_coords[0].x, B_coords[0].y, seg_cnt);// debug print to see how often fail of 1 occurs and passes anyways
		case 0:
			;
		}
*/	}
//	write_imagei(ii4_sparse_adj_matrix, indices, candidates.i);
}
