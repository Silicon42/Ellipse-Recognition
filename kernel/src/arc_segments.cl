#include "cast_helpers.cl"
#include "arc_data.cl"
#define ACCEL_THRESH 20.f
//NOTE: all memory accesses to the 2D texture are basically random on a work item level and have minimal 2D locality within the
// a single work item due to segments traversing the image and the majority may likely be cache misses, so they are kept to an
// absolute minimum

// normalization scaling factors for making angular accel calculations more accurate to euclidean distance traversed
constant const float accel_norm[] = {
	1.f,			// 1/sqrt(1^2 + 0^2)
	0.70710677f,	// 1/sqrt(1^2 + 1^2)		== 1/sqrt(2)
	0.8944272f,		// 1/sqrt(1^2 + 0.5^2)		== 2/sqrt(5)
	0.8944272f,		// 1/sqrt(1^2 + 0.5^2)		== 2/sqrt(5)
	1.4142135f,		// 1/sqrt(0.5^2 + 0.5^2)	== sqrt(2)
	1.f,			// 1/sqrt(1^2 + 0^2)
	2.f,			// 1/sqrt(0.5^2 + 0^2)
	2.f				// 1/sqrt(0.5^2 + 0^2)
};

kernel void arc_segments_alt(
	read_only image1d_t iS2_start_coords,
	read_only image2d_t uc1_cont_info,
	read_only image2d_t iC1_grad_ang,
	write_only image2d_t ui4_arc_data)
{
	union l_conv coords, base_coords, prev_coords;
	short index = get_global_id(0);	// must be scheduled as 1D

	// initialize variables of arcs segment tracing loop for first iteration
	coords.i = read_imagei(iS2_start_coords, index).lo;
	if(!coords.l)	// this does mean a start at (0,0) won't get processed but I don't think that's particularly likely to happen and be critical
		return;
	
	base_coords = coords;

	// the cached 0-deg-relative bin index for where the next pixel was found when finding segment starts
	uchar cont_data = read_imageui(uc1_cont_info, coords.i).x;	
	uchar prev_cont_idx = cont_data & 7;	// start specified in start_info is assumed to be valid and have start flag, so can be safely masked to just index
	uchar is_supported = cont_data & 0x10;	// there was another supporting pixel at the start

	char prev_angle, curr_angle, prev_angle_diff, curr_angle_diff;
	uchar cont_idx, accel_scale_idx;
	float angle_accel_total, max_accel, min_accel;
	prev_angle = read_imagei(iC1_grad_ang, coords.i).x;
	coords.i += offsets[prev_cont_idx];

	curr_angle = read_imagei(iC1_grad_ang, coords.i).x;

	curr_angle_diff = curr_angle - prev_angle;

	// we define arc segments as having a roughly constant angular rate of change along their path_length within some minimum noise floor,
	// this means that we want to track overall drift of this angular "acceleration" which we would like to stay near 0 and if
	// it deviates, then we need to start a new arc segment and reset the angular acceleration to be just based on the last 3 pixels?? might be slightly off with that logic
	angle_accel_total = max_accel = min_accel = 0;
	uchar path_hist[32];	//shift register that stores return data for what pixels were included in the calculation
	path_hist[0] = prev_cont_idx;
	uchar path_length = 1;
	int2 offset_mid = 0;
	cont_data = read_imageui(uc1_cont_info, coords.i).x;

	ushort watchdog = 0;

	// loop until we hit a start or run out of pixels for the arc segment
	while((cont_data & 0xC8) == 0x08)	// check start flag and forced end flag not set, and continuation flag set
	{
		++watchdog;
		cont_idx = cont_data & 7;	// clear validity to use as index
		prev_coords = coords;
		coords.i += offsets[cont_idx];
	
		// update angle tracking variables
		prev_angle = curr_angle;
		// read angle + continuation data from inputs for this pixel
		curr_angle = read_imagei(iC1_grad_ang, coords.i).x;
		cont_data = read_imageui(uc1_cont_info, coords.i).x;

		// update angle diff tracking variables
		prev_angle_diff = curr_angle_diff;
		curr_angle_diff = curr_angle - prev_angle;

		// select the scale factor corresponding to the distance traversed between the midpoints of the 2 connections
		// this has a slight smoothing effect that is much closer to the actual scale of the angular acceleration as
		// opposed to simply taking the integer difference of differences
		accel_scale_idx = (abs((char)((cont_idx - prev_cont_idx) << 5)) >> 4) | (cont_idx & 1);

		// update angle accel tracking variables
		angle_accel_total += (curr_angle_diff - prev_angle_diff) * accel_norm[accel_scale_idx];
		max_accel = angle_accel_total > max_accel ? angle_accel_total : max_accel;
		min_accel = angle_accel_total < min_accel ? angle_accel_total : min_accel;
		prev_cont_idx = cont_idx;

		// if the total angular acceleration for the arc exceeds the small acceleration threshold,
		// save the accumulator and restart it as a new arc segment
		if(max_accel - min_accel > ACCEL_THRESH)
		{
			angle_accel_total = max_accel = min_accel = 0;
			//Calculate flags and center and write to output
			write_arc_data(path_length, 0, ui4_arc_data, base_coords.i, prev_coords.i, offset_mid);
			// reset path_accum
			base_coords = prev_coords;
			path_length = 1;
			path_hist[0] = cont_idx;
			offset_mid = 0;
		}
		else
		{
			if(!(path_length & 1))	// on even count, advance the midpoint by one
			{
				offset_mid += offsets[path_hist[(path_length >> 1) & 0x1F]];
			}
			// if we have exceeded the maximum size we can store in a single write, we reset and continue with a new one
			if(path_length == 127)
			{
				//Calculate flags and center and write to output
				write_arc_data(path_length, 0, ui4_arc_data, base_coords.i, prev_coords.i, offset_mid);
				// reset path_accum
				base_coords = prev_coords;
				path_length = 1;
				path_hist[0] = cont_idx;
				offset_mid = 0;
			}
			else
			{
				if(path_length < 32)
					path_hist[path_length & 0x1F] = cont_idx;
				++path_length;
			}
		}
	}

	// if edge was a lone edge of no more than 5 pixels and has no additional supporting segments, reject as noise
	// 5 is chosen semi-arbitrarily because that's the minimum number of points to fit an ellipse
	if(watchdog <= 5 && !((cont_data & 8) || is_supported))	// right or left of the traversed edge was connected to separate processing
		return;

	write_arc_data(path_length, 1, ui4_arc_data, base_coords.i, coords.i, offset_mid);
}