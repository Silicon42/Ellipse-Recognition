
//NOTE: all memory accesses to the 2D texture are basically random on a work item level and have minimal 2D locality within the
// a single work item due to segments traversing the image and the majority may likely be cache misses, so they are kept to an
// absolute minimum
union ul2_ui4{
	ulong2 ul2;
	uint4 ui4;
};
/*
union i_c4{
	int i;
	char4 c;
	uchar4 uc;
	uchar uca[4];
};
*/
kernel void arc_segments(read_only image1d_t us4_start_info, read_only image2d_t uc1_starts_image, read_only image2d_t iC1_grad_image, write_only image2d_t us4_path_image, write_only image2d_t uc1_trace)
{
	const int2 offsets[8] = {(int2)(0,1),(int2)(-1,1),(int2)(-1,0),(int2)(-1,-1),(int2)(0,-1),(int2)(1,-1),(int2)(1,0),(int2)(1,1)};
	short index = get_global_id(0) + 1;	// must be scheduled as 1D
	ushort max_size = read_imageui(us4_start_info, 0).x;
	if(index >= max_size)	//prevent reading from unitialized memory
		return;
	
	// initialize variables of arcs segment tracing loop for first iteration
	uint4 start_info = read_imageui(us4_start_info, index);
	int2 base_coords = (int2)(start_info.x, start_info.y);
	uchar dir_idx = start_info.z & 7;	// the cached 0-deg-relative bin index for where the next pixel was found when finding segment starts

	int2 coords = base_coords;
	write_imageui(uc1_trace, coords, -1);

	char prev_angle, curr_angle, prev_angle_diff, curr_angle_diff, angle_accel;
	prev_angle = read_imagei(iC1_grad_image, coords).x & 0xFE;
	//char inc_flags = ((short)0b0000111000001110 >> dir_idx) & 0b01010101;	//TODO: compare this perf wise to lookup, reduced space may boost perf
	coords += offsets[dir_idx];
	curr_angle = read_imagei(iC1_grad_image, coords).x & 0xFE;

	prev_angle_diff = curr_angle_diff = curr_angle - prev_angle;
	// we define arc segments as having a roughly constant angular rate of change along their path_length within some minimum noise floor,
	// this means that we want to track overall drift of this angular "acceleration" which we would like to stay near 0 and if
	// it deviates, then we need to start a new arc segment and reset the angular acceleration to be just based on the last 3 pixels?? might be slightly off with that logic
	angle_accel = 0;
	union ul2_ui4 path_accum;
	path_accum.ul2.x = dir_idx;	//shift register that stores return data for what pixels were included in the calculation
	uchar path_length = 1;

	ushort watchdog = 0;
	char min_length_override = 0;

	// loop until we hit a start or run out of pixels for the arc segment
	while(!(read_imageui(uc1_starts_image, coords).x & 8))
	{
		++watchdog;
		if(!watchdog)
		{
			printf("watchdog: segment too long. Is it looped?: (%i, %i), curr_angle: %i, prev_angle: %i\n", coords.x, coords.y, curr_angle, prev_angle);
			break;
		}
		//printf("(%3v2i),", coords);
		write_imageui(uc1_trace, coords, -1);
		// convert the reported gradient angle to a 3 bit index prediction for the next pixel we expect to be populated
		// this is calculated by current angle + derivative + rounding to bin factor(256/16 == 16) + 90 offset for normal(256/4 == 64),
		// then right shifted so only the 3 most significant bits remain
		char predicted_angle = curr_angle;// + prev_angle_diff;// + 16 + 64;
		dir_idx = (uchar)(predicted_angle + 16) >> 5;	// literal needs to be forced to be interpreted as char or else index might overflow

		prev_angle = curr_angle;

		union i_c4 geusses, diffs;
		//geusses.i = diffs.i = 0;
		geusses.c.x = read_imagei(iC1_grad_image, coords + offsets[(dir_idx-1) & 7]).x;
		geusses.c.y = read_imagei(iC1_grad_image, coords + offsets[dir_idx & 7]).x;
		geusses.c.z = read_imagei(iC1_grad_image, coords + offsets[(dir_idx+1) & 7]).x;
		geusses.c.w = read_imagei(iC1_grad_image, coords + offsets[(dir_idx+1) & 7]).x;

		if(!geusses.i)	// this is an early out for if there is no continuation, but it could be caught at the diffs check instead
			break;	// no continuation, end loop and flush accumulated contents to output

		// create a mask for the occupied pixels in the geusses from their occupancy flags
		// this gets used to filter out invalid results of computations on unoccupied geusses
		int occupancy_mask = geusses.i & 0x01010101;
		occupancy_mask = (occupancy_mask << 8) - occupancy_mask;

		geusses.i &= 0xFEFEFE;	// remove occupancy flag before use

		diffs.uc = abs(geusses.c - predicted_angle);
		diffs.i |= ~occupancy_mask;	// forces invalid diffs to max value

		uchar best_diff = 3;

		for(uchar i = 0; i < 3; ++i)	// select the edge pixel whose gradient most aligns with the predicted angle
		{	//NOTE: slight assymetry due to evaluation order, could be fixed with unrolling and hardcoding the index
			best_diff = (diffs.uca[best_diff] > diffs.uca[i]) ? i : best_diff;
		}


		//printf("diff: %u\n", diffs.uca[best_diff]);
		// prevent jumping to completely differently aligned gradients, also prevents previously processed pixels from overlapping
		// the geuss region, which can lead to an infinite loop of going back and forth between the two pixels
		if(diffs.uca[best_diff] >= 32)
			break;	// none of the geusses were aligned with the current angle prediction, end loop and flush accumulated contents to output
		
		curr_angle = (char)geusses.uca[best_diff];
		dir_idx = (dir_idx + best_diff - 1) & 7;

		prev_angle_diff = curr_angle_diff;
		curr_angle_diff = curr_angle - prev_angle;
		if(dir_idx & 1)	// if it's a diagonal the diff is less due to being spread over a greater distance
			curr_angle_diff *= M_SQRT1_2_F;
		angle_accel += curr_angle_diff - prev_angle_diff;

		// if statements typically not taken so at any given time most work groups won't diverge

		// if the total angular acceleration for the arc exceeds the small acceleration threshold,
		// save the accumulator and restart it as a new arc segment
		if(abs(angle_accel) > 6)	// this corresponds to a +/- 2.8125 deg per pixel^2 total acceleration noise threshold
		{
			angle_accel = 0;
			//Write path_accum to 2D image
			write_imageui(us4_path_image, base_coords, path_accum.ui4);
			// reset path_accum
			base_coords = coords;
			path_length = 0;
			path_accum.ul2 = 0;
		}

		coords += offsets[dir_idx];
		path_length++;
		path_accum.ul2 = (path_accum.ul2.x << 3) | dir_idx;

		//if we've filled a 64-bit value copy it to the next one and continue
		if(path_length == 21)
		{
			path_accum.ul2.y = path_accum.ul2.x;
		}
		// if we have exceeded the maximum size we can store in a single write, we reset and continue with a new one
		else if(path_length >= 42)
		{
			//Write path_accum to 2D image
			write_imageui(us4_path_image, base_coords, path_accum.ui4);
			// reset path_accum
			base_coords = coords;
			path_length = 0;
			path_accum.ul2 = 0;
		}
	}

	// if edge was a lone edge of no more than 6 pixels, reject as noise
	if(watchdog < 5 )//&& !min_length_override)
	{
	//	printf(" length \n");
		return;
	}

	write_imageui(us4_path_image, base_coords, path_accum.ui4);
}