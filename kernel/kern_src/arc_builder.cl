#include "cast_helpers.cl"
#include "math_helpers.cl"
// fast contiguous segment elliptical arc classification

//FIXME: move this to a separate file for repeated use then come back and convert floats to floats where possible
/*
#ifndef float
#define float float
#define float2 float2
#endif
*/
constant const char order[8] = {0,1,2,3,0,2,1,3};

// calculates a ellipse through 5 points where 1 point is (0,0) and the rest are relative to it
// returns the foci coordinates, distance from foci to edge is implied
// if the conic through 5 points would not be an ellipse, returns NaN
float4 ellipse_from_hist(private const int2 diffs[4], private const int cross_prods[4])
{	//TODO: see how to mitigate rounding errors better
//if(all(diffs[0]==(int2)(75,-27)))
//	printf("%i	%i	%i	%i\n", cross_prods[0],cross_prods[1],cross_prods[2],cross_prods[3]);
//	printf("%v2i	%v2i	%v2i	%v2i\n", diffs[0],diffs[1],diffs[2],diffs[3]);
	float4 foci;
	float2 ca, ed, rs, temp_f2;
	float b, temp_f, inv_2t, ac_diff;
	float u, v;
	int2 temp_i2;

	// Fix to prevent exponent overflow from too many multiplication steps by pre-scaling the u and v values
	// technically it might be safer to divide by the avg exponent between the max and non-zero-min of the coefficients,
	// but dividing by a constant power of 2 is faster and should work in most cases, especially if resolution is kept
	// to reasonable values (ie roughly <= 4069)
	//FIXME: max guaranteed safe divisor with -cl-denorms-are-zero set is 2147483648 (2^31), need to add defines that take that into account
	u =  (cross_prods[1] * cross_prods[3]) / 137438953472.0f;	// bias exponent by dividing by 2^37, max safe value without losing fine resolution
	v = -(cross_prods[0] * cross_prods[2]) / 137438953472.0f;	// compiler should hopefully optimize this to simple exponent setting since it's a power of 2

	ca = u * convert_float2(diffs[0] * diffs[2]) + v * convert_float2(diffs[1] * diffs[3]);
	temp_i2 = diffs[0] * diffs[2].yx;
	b = -u * (float)(temp_i2.x + temp_i2.y);
	temp_i2 = diffs[1] * diffs[3].yx;
	b -= v * (float)(temp_i2.x + temp_i2.y);
//ca = (float2)(-40,-33);
//b=-24;
	inv_2t = (4 * ca.x * ca.y - b * b);
if(all(diffs[0]==(int2)(75,-27)))
printf("%e", inv_2t);
	//only bother computing foci for ellipse candidates, not parabolas or hyperbolas
	if(inv_2t <= 0)
		return NAN;
	
	inv_2t = 1 / inv_2t;

	ed = u * (cross_prods[0] * convert_float2(diffs[2]) + cross_prods[2] * convert_float2(diffs[0]))\
		+v * (cross_prods[1] * convert_float2(diffs[3]) + cross_prods[3] * convert_float2(diffs[1]));
if(all(diffs[0]==(int2)(75,-27)))
printf("%v2e", ca);
//ed=(float2)(168);
	char negate = all(ca < 0) ? -1:1;	// this is to prevent the temp_f value from going negative because the square root can't handle that
	b *= negate;
	ed *= (float2)(-negate, negate);
	ca *= negate;
//if(negate < 0)
//	printf("n");

	rs = b * ed;			//b[e, d]
	temp_f = -rs.x * ed.y;	//-bde
	temp_f2 = ca * ed.yx;	//[cd, ae]
	rs -= 2 * temp_f2;		//b[e, d] - 2[cd, ae]
	ac_diff = ca.y - ca.x;	//a-c

	temp_f = 2 * (temp_f + dot_2d_f(temp_f2, ed.yx));	//2(ae^2 - bde + cd^2)
	temp_f2 = sqrt(temp_f * (hypot(ac_diff, b) + (float2)(-ac_diff, ac_diff)));
if(any(isnan(temp_f2)))
	printf("X");

	// due to sqrt of complex value, x and y components are either same sign if b > 0 or opposite sign if b < 0
	if(b > 0)
		temp_f2.y *= -1;

	foci.lo = rs - temp_f2;
	foci.hi = rs + temp_f2;
	foci *= inv_2t;
//printf("%v4f ]\n", foci);
	
	return convert_float4(foci);
}

// adds the coefficient components as calculated for this point to the square matrix
// done in long int math to prevent precision loss, safe from overflow as long as
// no more than 256 points are added this way for max x and y coords < 16384 (2^14),
// safe limit is higher if max coords are less than that, considering the longest
// chain I've seen so far was ~15, I'm not even going to check
void addPointCoeffs(ulong4 coeffs[4], int2 p)
{
	ulong x2 = p.x * p.x;
	ulong y2 = p.y * p.y;
	ulong xy = p.x * p.y;
	//TODO: this could probably be done in a more efficient order, also the ordering
	// the ordering might not be ideal for later conversion to packed symmetric 
	// form from stored coefficients in terms of accuracy loss
	coeffs[0] += (ulong4)(x2, xy, y2, x2*p.x);
	coeffs[1] += (ulong4)(x2*p.y, x2*x2, p.x*y2, p.y*y2);
	coeffs[2] += (ulong4)(1, y2*y2, p.x, p.y);
	coeffs[3] += (ulong4)(x2*xy, xy*y2, x2*y2, 0);
}

inline float get_ellipse_dist(const float4 foci)
{
	return fast_length(foci.lo) + fast_length(foci.hi);
}

inline char is_near_ellipse_edge(const float4 foci, const float dist, const float2 point)
{
	return fabs(dist - (fast_distance(point, foci.lo) + fast_distance(point, foci.hi))) < 2;
}

kernel void arc_builder(
	read_only image1d_t is2_start_coords,
	read_only image2d_t ic2_line_data,
	read_only image1d_t us1_line_counts,
	write_only image2d_t us1_seg_in_arc,
	write_only image2d_t ff4_ellipse_foci,	//TODO: ff4_ellipse_foci is only used for debugging, remove it eventually
	write_only image2d_t ff4_pseudo_coeffs)	//contains the 12 unique coefficients that the self transpose product produces as part of calculating pseudo inverse
{
	short index = get_global_id(0);	// must be scheduled as 1D
/*
if(index)
	return;
int2 points[4] = {(int2)(-2,4),(int2)(5,7),(int2)(8,5),(int2)(10,1)};
int2 diffs[4];
int cross[4];
diffs[0] = points[0]-points[3];
diffs[1] = points[1]-points[0];
diffs[2] = points[2]-points[1];
diffs[3] = points[3]-points[2];
cross[0] = cross_2d_i(points[0], points[3]);
cross[1] = cross_2d_i(points[1], points[0]);
cross[2] = cross_2d_i(points[2], points[1]);
cross[3] = cross_2d_i(points[3], points[2]);
ellipse_from_hist(diffs, cross);
return;
*/
	// get count of line segments in this chain of processing
	int remaining_segs = read_imageui(us1_line_counts, index).x;

	// reject unpopulated starts
	if(!remaining_segs)
		return;

	// get starting pixel coordinates
	int2 base_coords = read_imagei(is2_start_coords, index).lo;

	ulong4 coeffs[4];	// accumulator for the 12 unique coeffs of the self-transpose-product matrix
	int2 total_offset, curr_coords, curr_seg, prev_seg;
	curr_coords = base_coords;
	private int2 points[4];	// relative points to last reset used in the 
	curr_seg = read_imagei(ic2_line_data, base_coords).lo;
	
	private int cross_prods[4];
	private int8 diffs8;
	private int2* diffs = (private void*)&diffs8;
	char reset = 3;
	ushort seg_cnt;
	float4 foci;
	float edge_dist;
	char dir, dir_trend;
	int dir_cross;
	uchar kick = 0;	// which point index to kick when a recalculation occurs

	// loop over all segments that came from this start
	// don't have to worry about returning to start b/c with the forward acute angle restriction
	// that would require at least 5 segments and therefore wouldn't end up with one of the points
	// as (0,0) on the initial calculation
	while(--remaining_segs)
	{
		switch(reset)
		{
		case 1:	// logical reset, last read segment can't be part of the same elliptical arc
			write_imageui(us1_seg_in_arc, base_coords, seg_cnt);
			// write coefficients out to buffer
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4, base_coords.y), convert_float4(coeffs[0]));
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+1, base_coords.y), convert_float4(coeffs[1]));
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+2, base_coords.y), convert_float4(coeffs[2]));
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+3, base_coords.y), convert_float4(coeffs[3]));

			//if it was long enough to calculate an ellipse, write out the foci
			if(seg_cnt >= 4)
			{
				float2 base_f = convert_float2(base_coords);
				foci += (float4)(base_f, base_f);
				write_imagef(ff4_ellipse_foci, base_coords, foci);
			}
			base_coords += total_offset;
		case 3:	// loop entry init/re-init
			reset = 0;
			coeffs[0] = coeffs[1] = coeffs[2] = 0;
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
			ulong4 base_coeffs[3] = {0};
			addPointCoeffs(base_coeffs, base_coords);
			coeffs[0] -= base_coeffs[0];
			coeffs[1] -= base_coeffs[1];
			coeffs[2] -= base_coeffs[2];
			addPointCoeffs(base_coeffs, new_base_coords);
			write_imageui(us1_seg_in_arc, base_coords, 1);
			// write the single segment coefficients scaled by 1/2 so as to not bias solutions toward single segments
			// other low segment counts may still cause biased weighting but not nearly as bad as single segments
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4, base_coords.y), convert_float4(base_coeffs[0])/2);
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+1, base_coords.y), convert_float4(base_coeffs[1])/2);
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+2, base_coords.y), convert_float4(base_coeffs[2])/2);
			write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+3, base_coords.y), convert_float4(base_coeffs[2])/2);
			
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
		
		dir = (dir_cross >= 0) ? dir_cross > 0 : -1;	//extract sign of dir_cross to get just the curving direction
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

				foci = ellipse_from_hist(diffs, cross_prods);

				// if points didn't form an ellipse
				if(!isfinite(foci.x))
				{
					reset = 2;
					continue;	//continue without advancing segment count
				}
				
				edge_dist = get_ellipse_dist(foci);
				float2 mid0 = convert_float2(points[0]) / 2;
				// if the ellipse was a bad fit, try again next time
				if(!is_near_ellipse_edge(foci, edge_dist, mid0))
				{
					reset = 2;
					continue;	//continue without advancing segment count
				}
			}
		}
		else
		{
			// if the new segment endpoint deviates from the already calculated ellipse,
			// it either needs to be re-calculated with the new point or reset and written out
			if(!is_near_ellipse_edge(foci, edge_dist, convert_float2(total_offset)))
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
				float4 new_foci = ellipse_from_hist(diffs, cross_prods);
				if(!isfinite(new_foci.x))
				{
					reset = 1;
					continue;
				}
				float new_dist = get_ellipse_dist(new_foci);
				// if the new calculation wouldn't include the old point, it needs to be written out and reset
				if(!is_near_ellipse_edge(new_foci, new_dist, old_point))
				{
					reset = 1;
					continue;
				}
				// else this was just a minor course correction and can be taken as the updated ellipse approx.
				foci = new_foci;
				edge_dist = new_dist;
			}
		}
		// this must stay at the end b/c some situations need to be able to skip it
		++seg_cnt;
	}
//	if(all(base_coords==(int2)(490,590)))
//		printf("%v2i	%v2i	%v2i	%v2i\n", points[0],points[1],points[2],points[3]);

	//flush last arc
	write_imageui(us1_seg_in_arc, base_coords, seg_cnt);
	write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4, base_coords.y), convert_float4(coeffs[0]));
	write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+1, base_coords.y), convert_float4(coeffs[1]));
	write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+2, base_coords.y), convert_float4(coeffs[2]));
	write_imagef(ff4_pseudo_coeffs, (int2)(base_coords.x*4+3, base_coords.y), convert_float4(coeffs[3]));

	//if it was long enough to calculate an ellipse, write out the foci
	if(seg_cnt >= 4)
	{
		float2 base_f = convert_float2(base_coords);
		foci += (float4)(base_f, base_f);
		write_imagef(ff4_ellipse_foci, base_coords, foci);
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
