#include "cast_helpers.cl_h"
#include "math_helpers.cl_h"
#include "conic_solve.cl_h"
// fast contiguous segment elliptical arc classification

//FIXME: move this to a separate file for repeated use then come back and convert floats to floats where possible
/*
#ifndef float
#define float float
#define float2 float2
#endif
*/
constant const char order[8] = {0,1,2,3,0,2,1,3};

//TODO: VVV this VVV value needs fine tuning
#define ELLIPSE_DEVIATION_THRESH 16

kernel void arc_builder(
	read_only image1d_t is2_start_coords,
	read_only image2d_t ic2_line_data,
	read_only image1d_t us1_line_counts,
	write_only image2d_t us1_seg_in_arc,
	write_only image2d_t ff4_ellipse_foci,	//TODO: ff4_ellipse_foci is only used for debugging, remove it eventually
	write_only image2d_t ff4_pseudo_coeffs)	//contains the 12 unique coefficients that the self transpose product produces as part of calculating pseudo inverse
{
	short index = get_global_id(0);	// must be scheduled as 1D

	// get count of line segments in this chain of processing
	int remaining_segs = read_imageui(us1_line_counts, index).x;

	// reject unpopulated starts
	if(!remaining_segs)
		return;

	// get starting pixel coordinates
	int2 base_coords = read_imagei(is2_start_coords, index).lo;

	ulong4 coeffs[4];	// accumulator for the 14+1 unique coeffs of the self-transpose-product matrix
	int2 total_offset, curr_coords, curr_seg, prev_seg;
	curr_coords = base_coords;
	private int2 points[4];	// relative points to last reset used in the 
	curr_seg = read_imagei(ic2_line_data, base_coords).lo;

	private int cross_prods[4];
	private int8 diffs8;
	private int2* diffs = (private void*)&diffs8;
	char reset = 3;
	ushort seg_cnt;
	Ellipse ellipse;
	float4 * foci = &ellipse.foci_dist.foci;
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
		case 1:	// logical reset, last read segment can't be part of the same elliptical arc
			write_imageui(us1_seg_in_arc, base_coords, seg_cnt);
			// write coefficients out to buffer
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4,   base_coords.y), convert_float4(coeffs[0]));
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+1, base_coords.y), convert_float4(coeffs[1]));
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+2, base_coords.y), convert_float4(coeffs[2]));
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+3, base_coords.y), convert_float4(coeffs[3]));

			//if it was long enough to calculate an ellipse, write out the foci
			if(seg_cnt >= 4)
			{
				float2 base_f = convert_float2(base_coords);
				*foci += (float4)(base_f, base_f);
				write_imagef(ff4_ellipse_foci, base_coords, *foci);
			}
			base_coords += total_offset;
		case 3:	// loop entry init/re-init
			reset = 0;
			coeffs[0] = coeffs[1] = coeffs[2] = coeffs[3] = 0;
			addPointCoeffs(coeffs, base_coords);
			total_offset = 0;	//keep last segment that caused the reset
			seg_cnt = 1;
			dir_trend = 0;	//trend unknown since only 1 segment at this point
			break;
		case 2:	// first solve reset, at time of adding 4th segment, failed to get a valid ellipse fit
			reset = 0;
			// kick first segment and copy things down 1 slot to try again
			int2 first_point = points[0];
			int2 new_base_coords = base_coords + first_point;
			ulong4 base_coeffs[4] = {0};
			addPointCoeffs(base_coeffs, base_coords);
			coeffs[0] -= base_coeffs[0];
			coeffs[1] -= base_coeffs[1];
			coeffs[2] -= base_coeffs[2];
			coeffs[3] -= base_coeffs[3];
			addPointCoeffs(base_coeffs, new_base_coords);
			write_imageui(us1_seg_in_arc, base_coords, 1);
			// write the single segment coefficients scaled by 1/2 so as to not bias solutions toward single segments
			// other low segment counts may still cause biased weighting but not nearly as bad as single segments
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4,   base_coords.y), convert_float4(base_coeffs[0])/2);
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+1, base_coords.y), convert_float4(base_coeffs[1])/2);
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+2, base_coords.y), convert_float4(base_coeffs[2])/2);
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+3, base_coords.y), convert_float4(base_coeffs[3])/2);
			
			base_coords = new_base_coords;	// advance base coords by first segment
			total_offset -= first_point;
			points[0] = points[1] - first_point;	// remove first segment's offset to account for new base coord
			points[1] = points[2] - first_point;
			points[2] = points[3] - first_point;
			diffs[0] = diffs[1];
			diffs[1] = diffs[2];
			diffs[2] = diffs[3];
		}
		prev_seg = curr_seg;
		total_offset += curr_seg;
		curr_coords += curr_seg;
		addPointCoeffs(coeffs, curr_coords);
		
		curr_seg = read_imagei(ic2_line_data, curr_coords).lo;

		// angle difference between segments A and B must be acute (no sharp corners), ie positive dot product
		int dir_dot = dot_2d_i(prev_seg, curr_seg);
		if(dir_dot <= 0)
		{
			reset = 1;	//set reset flag
			continue;
		}
		
		// angle between segments was more than 45 degrees
		dir_cross = cross_2d_i(prev_seg, curr_seg);
		if(abs(dir_cross) > dir_dot)
		{
			reset = 1;
			continue;
		}
		
		dir = (dir_cross >= 0) ? (dir_cross > 0) : -1;	//extract sign of dir_cross to get just the curving direction
		// if curving direction changes between +/- trigger a reset
		if((dir ^ dir_trend) == -2)
		{
			reset = 1;
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

				ellipse_from_hist(diffs, cross_prods, &ellipse);

				// if points didn't form an ellipse with a reasonable minimum major axis length
				if(ellipse.foci_dist.dist <= 2)
				{
					reset = 2;
					continue;	//continue without advancing segment count
				}
				
				float2 mid0 = convert_float2(points[0]) / 2;
				float deviation = get_ellipse_deviation(&ellipse.foci_dist, mid0);
				// if the ellipse was a bad fit, try again next time
				if(deviation > ELLIPSE_DEVIATION_THRESH)
				{
	//				printf("seg_cnt3: %f", ellipse.foci_dist.dist);
					reset = 2;
					continue;	//continue without advancing segment count
				}
			}
		}
		else
		{
	//				printf("test1 ");
			// if the new segment endpoint deviates from the already calculated ellipse,
			// it either needs to be re-calculated with the new point or reset and written out
			if(get_ellipse_deviation(&ellipse.foci_dist, convert_float2(total_offset)) > ELLIPSE_DEVIATION_THRESH)
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
				Ellipse new_ellipse;
				ellipse_from_hist(diffs, cross_prods, &new_ellipse);
				if(new_ellipse.foci_dist.dist <= 0)
				{
					reset = 1;
					continue;
				}

		//			printf("test2 ");
				// if the new calculation wouldn't include the old point, it needs to be written out and reset
				if(get_ellipse_deviation(&ellipse.foci_dist, old_point) > ELLIPSE_DEVIATION_THRESH)
				{
					reset = 1;
					continue;
				}
				// else this was just a minor course correction and can be taken as the updated ellipse approx.
				//TODO: make this a reference copy instead of a value copy
				ellipse = new_ellipse;
			}
		}
		// this must stay at the end b/c some situations need to be able to skip it
		++seg_cnt;
	} while(--remaining_segs);
//	if(all(base_coords==(int2)(490,590)))
//		printf("%v2i	%v2i	%v2i	%v2i\n", points[0],points[1],points[2],points[3]);

	//flush last arc
	write_imageui(us1_seg_in_arc, base_coords, seg_cnt);
	write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4,   base_coords.y), convert_float4(coeffs[0]));
	write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+1, base_coords.y), convert_float4(coeffs[1]));
	write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+2, base_coords.y), convert_float4(coeffs[2]));
	write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+3, base_coords.y), convert_float4(coeffs[3]));

	//if it was long enough to calculate an ellipse, write out the foci
	if(seg_cnt >= 4)
	{
		float2 base_f = convert_float2(base_coords);
		*foci += (float4)(base_f, base_f);
//printf("%v4f ]\n", *foci);
		write_imagef(ff4_ellipse_foci, base_coords, *foci);
	}
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
