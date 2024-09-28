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

kernel void arc_segments(
	read_only image1d_t iS2_start_coords,
	read_only image2d_t uc1_cont_info,
	read_only image2d_t iC1_grad_ang,
	write_only image2d_t ui4_arc_data)
{
	short index = get_global_id(0);	// must be scheduled as 1D
	//if(!index)
	//	printf("arc_segments entry\n");
	
	int2 coords, prev_coords;	// current and previous pixel coordinates
	struct arc_AB_tracking arcs = {0};

	// initialize variables of arcs segment tracing loop for first iteration
	coords = read_imagei(iS2_start_coords, index).lo;
	if(!((union l_conv)coords).l)	// this does mean a start at (0,0) won't get processed but I don't think that's particularly likely to happen and be critical
		return;
	
	arcs.base_coords[0] = coords;

	// the offset lookup index for where the next pixel was found when finding segment starts
	uchar cont_data = read_imageui(uc1_cont_info, coords).x;
	// start specified in start_info implicitly has a valid continuation, so can be safely masked to just index
	uchar prev_cont_idx = cont_data & 7;
	uchar is_supported = cont_data & 0x10;	// there was another supporting pixel at the start

	char first_angle, prev_angle, curr_angle, prev_angle_diff, curr_angle_diff;
	uchar cont_idx, accel_scale_idx;
	float angle_accel_total, max_accel, min_accel;

	// initialize angle, angle difference, and angle acceleration variables for loop entry
	first_angle = prev_angle = read_imagei(iC1_grad_ang, coords).x;
	coords += offsets[prev_cont_idx];
	curr_angle = read_imagei(iC1_grad_ang, coords).x;
	curr_angle_diff = curr_angle - prev_angle;
	// we define arc segments as having a roughly constant angular rate of change along their path_length within some minimum noise floor,
	// this means that we want to track overall drift of this angular "acceleration" which we would like to stay near 0 and if
	// it deviates, then we need to start a new arc segment and reset the angular acceleration to be just based on the last 3 pixels?? might be slightly off with that logic
	angle_accel_total = max_accel = min_accel = 0;
	uchar path_hist[32];	//shift register that stores return data for what pixels were included in the calculation
	path_hist[0] = prev_cont_idx;
	arcs.data[0].flags = IS_NOT_END;
	char* path_length = &arcs.data[0].len;
	*path_length = 1;
	cont_data = read_imageui(uc1_cont_info, coords).x;
//	ushort watchdog = 0;

	// loop until we hit a start or run out of pixels for the arc segment
	while((cont_data & 0xC8) == 0x08)	// check start flag and forced end flag not set, and continuation flag set
	{
	/*	++watchdog;
		if(!watchdog)
		{
			//printf("Looped\n");
			return;
		}
	*/	cont_idx = cont_data & 7;	// clear validity to use as index
		prev_coords = coords;
		coords += offsets[cont_idx];
	
		// update angle tracking variables
		prev_angle = curr_angle;
		// read angle + continuation data from inputs for this pixel
		curr_angle = read_imagei(iC1_grad_ang, coords).x;
		cont_data = read_imageui(uc1_cont_info, coords).x;

		// update angle diff tracking variables
		prev_angle_diff = curr_angle_diff;
		curr_angle_diff = curr_angle - prev_angle;

		// select the scale factor corresponding to the distance traversed between the midpoints of the 2 connections
		// this has a slight smoothing effect that is much closer to the actual scale of the angular acceleration as
		// opposed to simply taking the integer difference of differences
		accel_scale_idx = (abs((char)((cont_idx - prev_cont_idx) << 5)) >> 4) | (cont_idx & 1);
		prev_cont_idx = cont_idx;

		// update angle accel tracking variables
		angle_accel_total += (curr_angle_diff - prev_angle_diff) * accel_norm[accel_scale_idx];
		max_accel = angle_accel_total > max_accel ? angle_accel_total : max_accel;
		min_accel = angle_accel_total < min_accel ? angle_accel_total : min_accel;

		// on odd count, update the midpoint by one position in the path history
		if(*path_length & 1)
		{
			arcs.data[arcs.curr].offset_mid += offsets_c[path_hist[(*path_length >> 1) & 0x1F]];
		}

	//FIXME: some nasty divergent code here, can probably be inverted s.t. threads stop processing once they
	// should write and wait for other threads to get to their write calls or exit the loop and then write in parallel
	// path_length would likely drive an inner loop then with break outs for acceleration
	//FIXME: also while you're at it add an additional break out condition for if the first angle is more than 135 deg
	// from the curr angle and logic to reset the first_angle on write
		// if the total angular acceleration for the arc exceeds the small acceleration threshold,
		// save the accumulator and restart it as a new arc segment
		if(max_accel - min_accel > ACCEL_THRESH)
		{
			angle_accel_total = max_accel = min_accel = 0;
			//Calculate flags and center and write to output
			write_arc_data(ui4_arc_data, &arcs, prev_coords);
			// reset path tracking variables
			path_hist[0] = cont_idx;
			path_length = &arcs.data[arcs.curr].len;
			continue;
		}
		//else
		// if we have exceeded maximum size we can store in a single write, we reset and continue with a new one
		if(*path_length == 127)
		{
			//--*path_length;
			//Calculate flags and center and write to output
			write_arc_data(ui4_arc_data, &arcs, prev_coords);
			// reset path tracking variables
			path_hist[0] = cont_idx;
			path_length = &arcs.data[arcs.curr].len;
		}
		else
		{
			if(*path_length < 64)
				path_hist[*path_length & 0x1F] = cont_idx;

			++*path_length;
		}
	}

	// if edge was a lone edge of no more than 5 pixels and has no additional supporting segments, reject as noise
	// 5 is chosen semi-arbitrarily because that's the minimum number of points to fit an ellipse
	if(!(arcs.data[1].len) && arcs.data[0].len < 4 && !((cont_data & 8) || is_supported))	// right or left of the traversed edge was connected to separate processing
		return;

	//clear not end flag
	arcs.data[arcs.curr].flags &= ~IS_NOT_END;
	write_arc_data(ui4_arc_data, &arcs, prev_coords);
}