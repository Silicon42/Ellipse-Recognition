
//NOTE: all memory accesses to the 2D texture are basically random on a work item level and have minimal 2D locality within the
// a single work item due to segments traversing the image and the majority may likely be cache misses, so they are kept to an
// absolute minimum
union ul2_ui4{
	ulong2 ul2;
	uint4 ui4;
};

kernel void arc_segments(read_only image1d_t us4_start_info, read_only image2d_t uc1_starts_image, read_only image2d_t iC1_grad_image, write_only image2d_t us4_path_image, write_only image2d_t uc1_trace)
{
	const int2 offsets[8] = {(int2)(1,0),(int2)(1,1),(int2)(0,1),(int2)(-1,1),(int2)(-1,0),(int2)(-1,-1),(int2)(0,-1),(int2)(1,-1)};
	short index = get_global_id(0);	// must be scheduled as 1D
	
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

	// loop until we run out of pixels for the arc segment or hit a start
	while(!read_imageui(uc1_starts_image, coords).x)
	{
		write_imageui(uc1_trace, coords, -1);
		// convert the reported gradient angle to a 3 bit index prediction for the next pixel we expect to be populated
		// this is calculated by current angle + derivative + rounding to bin factor(256/16 == 16) + 90 offset for normal(256/4 == 64),
		// then right shifted so only the 3 most significant bits remain
		char predicted_angle = curr_angle + prev_angle_diff + 16 + 64;
		dir_idx = (uchar)predicted_angle >> 5;	// literal needs to be forced to be interpreted as char or else index might overflow

		prev_angle = curr_angle;
		curr_angle = read_imagei(iC1_grad_image, coords + offsets[dir_idx]).x;
		if(!curr_angle)
		{	// the guessed offsets get modified to the 2nd closesest pixel to the predicted angle
			//break;
			char dir = ((char)(dir_idx << 5) - predicted_angle) < 0 ? -1 : 1;
			dir_idx = (dir_idx + dir) & 7;	// & 7 to keep index in 0-7 range
			curr_angle = read_imagei(iC1_grad_image, coords + offsets[dir_idx]).x;
			if(!curr_angle)
			{
				//3rd guess is on the opposite side of the first guess
				dir_idx = (dir_idx - 2*dir) & 7;	// & 7 to keep index in 0-7 range
				curr_angle = read_imagei(iC1_grad_image, coords + offsets[dir_idx]).x;
				if(!curr_angle)
					break;	// no continuation, end loop and flush accumulated contents to output
			}
		}
		curr_angle &= 0xFE;	// remove occupancy flag before use

		prev_angle_diff = curr_angle_diff;
		curr_angle_diff = curr_angle - prev_angle;
		angle_accel += curr_angle_diff - prev_angle_diff;

		// if statements typically not taken so at any given time most work groups won't diverge

		// if the total angular acceleration for the arc exceeds the small acceleration threshold,
		// save the accumulator and restart it as a new arc segment
		if(abs(angle_accel) > 8)	// this corresponds to a +/- 2.8125 deg per pixel^2 total acceleration noise threshold
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
		if(path_length >= 42)
		{
			break;
			//Write path_accum to 2D image
			write_imageui(us4_path_image, base_coords, path_accum.ui4);
			// reset path_accum
			base_coords = coords;
			path_length = 0;
			path_accum.ul2 = 0;
		}
	}

	write_imageui(us4_path_image, base_coords, path_accum.ui4);
}