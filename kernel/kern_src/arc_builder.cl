#include "cast_helpers.cl_h"
#include "math_helpers.cl_h"
#include "conic_solve.cl_h"
#include "arc_data.cl_h"
// fast contiguous segment elliptical arc classification
//FIXME: reported seg_cnt seems to be one to high in some(?) cases
//FIXME: move this to a separate file for repeated use then come back and convert floats to floats where possible
/*
#ifndef float
#define float float
#define float2 float2
#endif
*/
constant const char order[8] = {0,1,2,3,0,2,1,3};

//TODO: See "Arc Adjacency Matrix-Based Fast Ellipse Detection" by Meng et al.
// They have an interesting way of categorizing segments as elliptical that might be
// more efficient than what I'm currently using if I can wrap my head around the math

//TODO: VVV this VVV value needs fine tuning
#define ELLIPSE_DEVIATION_THRESH 16

#define LOGICAL_RESET		1
#define LOOP_ENTRY_RESET	2
#define FIRST_SOLVE_RESET	3

void write_arc(
	write_only image2d_t ii2_arc_data, 
	write_only image2d_t is1_dir_cnt, 
	write_only image2d_t ff4_pre_solve_coeffs,
	write_only image2d_t ff4_ellipse_foci,
	write_only image2d_t ff1_ellipse_major,
	ulong16 * const coeffs,
	ArcData data,	// expected to have .tangents.lo pre-populated
	int2 const start_coords, 
	int2 const end_coords, 
	int2 const last_seg, //TODO: here on down might be more performant to combine into destination structs BEFORE passing them in
	int const dir, 
	int const seg_cnt,
	float const len_approx)
{
	write_imagei(is1_dir_cnt, start_coords, dir << 14 | min(seg_cnt, SEG_CNT_MASK));
	//TODO: see if packing a struct and writing the full width would be faster or if using the default alignment of write_image is faster
	
	data.tangents.hi = convert_char2(last_seg);
	data.endpoint = convert_short2(end_coords);

	write_imagei(ii2_arc_data, start_coords, (int4)(((RW_ArcData)data).rw, 0, 0));
//TODO: HIGH PRIORITY, solver needs to be able to handle hyperbolas b/c sometimes the fast method will pass arcs that are on the edge and
// their point data will go one way while the fast method goes the other
	// fix endpoint weights so they count for half as much as interior points, this prevents double weighting on shared endpoints
/*	if(seg_cnt > 1)
	{
		*coeffs *= 2;
		*coeffs -= getPointCoeffs(end_coords);
	}
*/	*coeffs += getPointCoeffs(start_coords);

	float16 coeffs_f = convert_float16(*coeffs);
	coeffs_f.s8 = len_approx;
	write_imagef(ff4_pre_solve_coeffs, (int2)(start_coords.x*2,   start_coords.y*2  ), coeffs_f.lo.lo);
	write_imagef(ff4_pre_solve_coeffs, (int2)(start_coords.x*2+1, start_coords.y*2  ), coeffs_f.lo.hi);
	write_imagef(ff4_pre_solve_coeffs, (int2)(start_coords.x*2,   start_coords.y*2+1), coeffs_f.hi.lo);
	write_imagef(ff4_pre_solve_coeffs, (int2)(start_coords.x*2+1, start_coords.y*2+1), coeffs_f.hi.hi);
	//TODO: there might be a perf benefit to not doing the rest if < 4 segments
	Conic conic;
	solveConic((private float*)&coeffs_f, conic.general);
	convertConicGeneralToFociMajor(&conic);
//	if(seg_cnt >= 4)
//		printf("%v4f		%.f\n", el.fm.foci, conic.fm.major);
//printf("%v2i	", start_coords);
	write_imagef(ff4_ellipse_foci, start_coords, conic.fm.foci);
	write_imagef(ff1_ellipse_major, start_coords, conic.fm.major);
}

kernel void arc_builder(
	read_only image1d_t is2_start_coords,	// branch-free segment chain processing starting locations
	read_only image2d_t ic2_line_data,		// displacement vectors for segment endpoints
	read_only image1d_t us1_line_counts,	// how many segments the corresponding start has associated with it, used to evaluate the correct number of segments
	write_only image2d_t ii2_arc_data,		// Contains first and last segment vector (tangent approx.) and end coordinates
	write_only image2d_t is1_dir_cnt,		// Contains signed direction in top 2 bits and clamped segment count in bottom 14 bits
//TODO: prevent double counting of the coefficients for endpoints by weighting them per segment rather than per point
// this might mean that these calculations would be better done in line segments than here
	write_only image2d_t ff4_pre_solve_coeffs,	// contains the 14 unique coefficients that the self transpose product produces as part of calculating pseudo inverse and a copy of the seg_cnt so that added coefficients know how many went into them
	write_only image2d_t ff4_ellipse_foci,	// Contains the foci as calculated according to the coefficients for arc segments of at least 4 segments
	write_only image2d_t ff1_ellipse_major)	// Contains the major axis length (foci-edge-foci length) as calculated according to the coefficients for arc segments of at least 4 segments
{
	short index = get_global_id(0);	// must be scheduled as 1D

	// get count of line segments in this chain of processing
	int remaining_segs = read_imageui(us1_line_counts, index).x;

	// reject unpopulated starts
	if(!remaining_segs)
		return;

	// get starting pixel coordinates
	int2 base_coords = read_imagei(is2_start_coords, index).lo;

	ulong16 coeffs;	// accumulator for the 14+1 unique coeffs of the self-transpose-product matrix
	// stores a running total of the coefficients as calculated for each point that is currently associated with the arc
	// EXCEPT the start point, which gets added at write time, this is because both end points of the arc are half the weight
	// for the solution so that when arcs that are separate but share endpoints are processed, they don't double up on the 
	// weighting in the solution, or more accurately all the other weights than the start point get doubled at write time,
	// then one instance of the start weights gets added and one instance of the end weights gets subtracted and if 2 arcs 
	// share the same point the sum of their weights will include exactly 2 copies of the same set of weights

	int2 total_offset, curr_coords, curr_seg, prev_seg;
	curr_coords = base_coords;
	private int2 points[4];	// relative points to last reset used in the 
	curr_seg = read_imagei(ic2_line_data, base_coords).lo;

	private int cross_prods[4];
	private int8 diffs8;
	private int2* diffs = (private void*)&diffs8;
	char reset = LOOP_ENTRY_RESET;
	ushort seg_cnt;
	float len_approx;
	ArcData data;
	Conic conic;
	float4 * foci = &conic.fm.foci;
	char dir, dir_trend;
	int dir_cross;
	uchar kick = 0;	// which point index to kick when a recalculation occurs

	// loop over all segments that came from this start
	// don't have to worry about returning to start b/c with the forward acute angle restriction
	// that would require at least 5 segments and therefore wouldn't end up with one of the points
	// as (0,0) on the initial calculation
	do
	{
		switch(reset)
		{
		case LOGICAL_RESET:	// last read segment likely can't be part of the same elliptical arc due to failing a logical test
			// write coefficients out to buffer
			write_arc(ii2_arc_data, is1_dir_cnt, ff4_pre_solve_coeffs, ff4_ellipse_foci, ff1_ellipse_major, &coeffs, data, base_coords, curr_coords, prev_seg, dir_trend, seg_cnt, len_approx);
			base_coords += total_offset;
		//	printf("%v16lu\n", coeffs);
			// intentional fall-through to re-init
		case LOOP_ENTRY_RESET:	// loop entry init/re-init
			reset = 0;
			coeffs = 0;
			total_offset = 0;	//keep last segment that caused the reset	//TODO: check if this comment still true
			seg_cnt = 1;
			len_approx = 0;
			data.tangents.lo = convert_char2(curr_seg);
			dir_trend = 0;	//trend unknown since only 1 segment at this point
			break;
		case FIRST_SOLVE_RESET:	// at time of adding 4th segment, failed to get a valid ellipse fit
			reset = 0;
			// kick first segment and copy things down 1 slot to try again
			int2 first_point = points[0];
			int2 new_base_coords = base_coords + first_point;
			ulong16 base_coeffs = getPointCoeffs(new_base_coords);
			coeffs -= base_coeffs;
			float first_len = mag_2d_i(diffs[0]);
			len_approx -= first_len;

			// write the single segment out
			write_arc(ii2_arc_data, is1_dir_cnt, ff4_pre_solve_coeffs, ff4_ellipse_foci, ff1_ellipse_major, &base_coeffs, data, base_coords, new_base_coords, new_base_coords-base_coords, 0, 1, first_len);
			
			base_coords = new_base_coords;	// advance base coords by first segment
			data.tangents.lo = convert_char2(diffs[1]);
			total_offset -= first_point;
			points[0] = points[1] - first_point;	// remove first segment's offset to account for new base coord
			points[1] = points[2] - first_point;
			points[2] = points[3] - first_point;
			diffs[0] = diffs[1];
			diffs[1] = diffs[2];
			diffs[2] = diffs[3];
		}
		total_offset += curr_seg;
		curr_coords += curr_seg;
		len_approx = mag_2d_i(curr_seg);
		coeffs += getPointCoeffs(curr_coords);
		prev_seg = curr_seg;
		curr_seg = read_imagei(ic2_line_data, curr_coords).lo;

		// angle difference between segments A and B must be acute (no sharp corners), ie positive dot product
		int dir_dot = dot_2d_i(prev_seg, curr_seg);
		if(dir_dot <= 0)
		{
			reset = LOGICAL_RESET;	//set reset flag
			continue;
		}
		
		// angle between segments was more than 45 degrees
		dir_cross = cross_2d_i(prev_seg, curr_seg);
		if(abs(dir_cross) > dir_dot)
		{
			reset = LOGICAL_RESET;
			continue;
		}
		
		dir = (dir_cross >= 0) ? (dir_cross > 0) : -1;	//extract sign of dir_cross to get just the curving direction
		// if curving direction changes between +/- trigger a reset
		if((dir ^ dir_trend) == -2)
		{
			reset = LOGICAL_RESET;
			continue;
		}
		// if curving direction hasn't yet collapsed to +/-1, attempt to do so
		dir_trend |= dir;
		
		// if we have not yet added enough segments to compute an ellipse
		if(seg_cnt <= 3)
		{
			// add them to the calculation cache
			points[seg_cnt-1] = total_offset;
			diffs[seg_cnt] = curr_seg;
		
			//on the attempt to add the 4th segment (currently has 3 segments)
			// we finally have enough points to attempt calculating the ellipse
			if(seg_cnt == 3)
			{
				points[3] = total_offset + curr_seg;
				//attempt to solve for ellipse and check if first segment matches
				diffs[0] = points[0] - points[3];
				cross_prods[0] = cross_2d_i(points[0], points[3]);
				cross_prods[1] = cross_2d_i(points[1], points[0]);
				cross_prods[2] = cross_2d_i(points[2], points[1]);
				cross_prods[3] = cross_2d_i(points[3], points[2]);
			//	printf("%v2i %v2i %v2i %v2i\n", diffs[0], diffs[1], diffs[2], diffs[3]);

			//	conic_from_hist(diffs, cross_prods, &conic);
				//FIXME: TEMPORARY SWAP OUT FOR ABOVE LINE FOR SANITY CHECKING (*1)
				float16 coeffs_f = convert_float16(coeffs);
				solveConic((__private float *)&coeffs_f, conic.general);
				convertConicGeneralToFociMajor(&conic);
				printf("%v4f\n", conic.fm.foci);
				//FIXME: (*1)*/

				// if points didn't form an ellipse with a reasonable minimum major axis length
				if(conic.fm.major <= 2)
				{
					reset = FIRST_SOLVE_RESET;
					continue;	//continue without advancing segment count
				}
				
				float2 mid0 = convert_float2(points[0]) / 2;
				float deviation = get_conic_deviation(&conic.fm, mid0);
				// if the ellipse was a bad fit, try again next time
				if(deviation > ELLIPSE_DEVIATION_THRESH)
				{
	//				printf("seg_cnt3: %f", ellipse.fm.major);
					reset = FIRST_SOLVE_RESET;
					continue;	//continue without advancing segment count
				}
			}
		}
		else
		{
	//				printf("test1 ");
			// if the new segment endpoint deviates from the already calculated ellipse,
			// it either needs to be re-calculated with the new point or reset and written out
			if(get_conic_deviation(&conic.fm, convert_float2(total_offset)) > ELLIPSE_DEVIATION_THRESH)
			{
				// lookup which entry to kick to attempt a re-calculation of the ellipse
				// the ordering is chosen so that it should spread the points out as recaluclations occur
				char k = order[kick++];
				kick &= 7;
				char k_m1 = (k - 1) & 3;
				char k_p1 = (k + 1) & 3;
				float2 old_point = convert_float2(points[k]);
				points[k] = total_offset;
				diffs[k] = points[k] - points[k_m1];
				diffs[k_p1] =  points[k_p1] - points[k];
				cross_prods[k] = cross_2d_i(points[k], points[k_m1]);
				cross_prods[k_p1] = cross_2d_i(points[k_p1], points[k]);

				// calculate the ellipse with the new point
				Conic new_conic;
				conic_from_hist(diffs, cross_prods, &new_conic);
				if(new_conic.fm.major <= 0)
				{
					reset = LOGICAL_RESET;
					continue;
				}

		//			printf("test2 ");
				// if the new calculation wouldn't include the old point, it needs to be written out and reset
				if(get_conic_deviation(&conic.fm, old_point) > ELLIPSE_DEVIATION_THRESH)
				{
					reset = LOGICAL_RESET;
					continue;
				}
				// else this was just a minor course correction and can be taken as the updated ellipse approx.
				//TODO: make this a reference copy instead of a value copy
				conic = new_conic;
			}
		}
		// this must stay at the end b/c some situations need to be able to skip it
		++seg_cnt;
	} while(--remaining_segs);
//	if(all(base_coords==(int2)(490,590)))
//		printf("%v2i	%v2i	%v2i	%v2i\n", points[0],points[1],points[2],points[3]);

	//flush last arc
	write_arc(ii2_arc_data, is1_dir_cnt, ff4_pre_solve_coeffs, ff4_ellipse_foci, ff1_ellipse_major, &coeffs, data, base_coords, curr_coords, prev_seg, dir_trend, seg_cnt, len_approx);

/*
	//if it was long enough to calculate an ellipse, write out the foci
	if(seg_cnt >= 4)
	{
		float2 base_f = convert_float2(base_coords);
		*foci += (float4)(base_f, base_f);
//printf("%v4f ]\n", *foci);
		write_imagef(ff4_ellipse_foci, base_coords, *foci);
	}*/
}

//debugging print stubs

/*/if(diffs[0].x == -145)
printf(
"base:	(%i,  %i)\n\
points:	(%i,  %i)	(%i,  %i)	(%i,  %i)	(%i,  %i)\n\
diffs:	(%i,  %i)	(%i,  %i)	(%i,  %i)	(%i,  %i)\n\
cross_prods:	%i	%i	%i	%i\n\
foci:	(%f, %f) (%f, %f)\n\n",\
base_coords,\
points[0], points[1], points[2], points[3],\
diffs[0], diffs[1], diffs[2], diffs[3],\
cross_prods[0], cross_prods[1], cross_prods[2], cross_prods[3],\
foci.x, foci.y, foci.z, foci.w);//*/

/*/if(diffs[0].x == -145)
{
printf(\
"diffs:	(%i,  %i)	(%i,  %i)	(%i,  %i)	(%i,  %i)\n\
cross_prods:	%i	%i	%i	%i\n\
u: %g,	v: %g\n\
a: %g	b: %g	c: %g	d: %g	e: %g\n\
r: %g	s: %g	2t: %g	tmp: %g	ac_diff: %g\n\n",\
diffs[0], diffs[1], diffs[2], diffs[3],\
cross_prods[0], cross_prods[1], cross_prods[2], cross_prods[3],\
u, v, ca.y, b, ca.x, ed.y, ed.x,\
 rs.x, rs.y, 1/inv_2t, temp_f, ac_diff);
}//*/
