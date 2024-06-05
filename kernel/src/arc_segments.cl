
//NOTE: all memory accesses to the 2D texture are basically random on a work item level and have minimal 2D locality within the
// a single work item due to segments traversing the image and the majority may likely be cache misses, so they are kept to an
// absolute minimum
union ul2_ui4{
	ulong2 ul2;
	uint4 ui4;
};

kernel void arc_segments(read_only image1d_t us4_src_image, read_only image2d_t iS4_grad_image, read_only image2d_t uc1_starts_image, write_only image2d_t ui4_dst_image)
{
	const int2 offsets[8] = {(int2)(1,0),(int2)(1,1),(int2)(0,1),(int2)(-1,1),(int2)(-1,0),(int2)(-1,-1),(int2)(0,-1),(int2)(1,-1)};
	short index = get_global_id(0);	// must be scheduled as 1D
	
	// initialize variables of arcs segment tracing loop for first iteration
	int4 base_coords = read_imagei(us4_src_image, index);
	int2 coords = base_coords.lo;
	char dir_idx = base_coords.z >> 5;	// the cached 0-deg-relative bin index for where the next pixel was found when finding segment starts

	short prev_angle, curr_angle, prev_angle_diff, curr_angle_diff, angle_accel;
	prev_angle = read_imagei(iS4_grad_image, coords).z;
	//char inc_flags = ((short)0b0000111000001110 >> dir_idx) & 0b01010101;	//TODO compare this perf wise to lookup, reduced space may boost perf
	coords += offsets[dir_idx];
	curr_angle = read_imagei(iS4_grad_image, coords).z;

	prev_angle_diff = curr_angle_diff = curr_angle - prev_angle;
	// we define arc segments as having a roughly constant angular rate of change along their path_length within some minimum noise floor,
	// this means that we want to track overall drift of this angular "acceleration" which we would like to stay near 0 and if
	// it deviates, then we need to start a new arc segment and reset the angular acceleration to be just based on the last 3 pixels?? might be slightly off with that logic
	angle_accel = 0;
	ulong2 path_accum;
	path_accum.x = base_coords.z & 7;	//shift register that stores return data for what pixels were included in the calculation
	uchar path_length = 1;

	do	// loop until we run out of pixels for the arc segment or hit a start
	{
		// convert the reported gradient angle to a 3 bit index prediction for the next pixel we expect to be populated
		// this is calculated by current angle + derivative + rounding to bin factor(65536/16 == 4096) + 90 offset for normal(65536/4 == 16384),
		// then right shifted so only the 3 most significant bits remain
		short predicted_angle = curr_angle + prev_angle_diff + 4096 + 16384;
		dir_idx = (ushort)predicted_angle >> 13;	// literal needs to be forced to be interpreted as short or else index might overflow

		prev_angle = curr_angle;
		curr_angle = read_imagei(iS4_grad_image, coords + offsets[dir_idx]).z;
		if(!curr_angle)
		{	// the guessed offsets get modified to the 2nd closesest pixel to the guess first
			char second_guess = predicted_angle - ((short)dir_idx << 13) >= 0 ? 1 : -1;
			dir_idx = (dir_idx + second_guess) & 7;	// & 7 to keep index in 0-7 range
			curr_angle = read_imagei(iS4_grad_image, coords + offsets[dir_idx]).z;
			if(!curr_angle)
			{
				//3rd guess is on the opposite side of the first guess
				dir_idx = (dir_idx - 2*second_guess) & 7;	// & 7 to keep index in 0-7 range
				curr_angle = read_imagei(iS4_grad_image, coords + offsets[dir_idx]).z;
				if(!curr_angle)
					break;	// no continuation, end loop and flush accumulated contents to output
			}
		}

		prev_angle_diff = curr_angle_diff;
		curr_angle_diff = curr_angle - prev_angle;
		angle_accel += curr_angle_diff - prev_angle_diff;

		// if statements typically not taken so at any given time most work groups won't diverge

		// if the total angular acceleration for the arc exceeds the small acceleration threshold,
		// save the accumulator and restart it as a new arc segment
		if(abs(angle_accel) > 64)
		{
			angle_accel = 0;
			//Write path_accum to 2D image
			write_imageui(ui4_dst_image, coords, ((union ul2_ui4)path_accum).ui4);
			// reset path_accum
			path_length = 0;
			path_accum = 0;
		}

		coords += offsets[dir_idx];
		path_length++;
		path_accum = (path_accum.x << 3) | dir_idx;

		//if we've filled a 64-bit value copy it to the next one and continue
		if(path_length == 21)
		{
			path_accum.y = path_accum.x;
		}
		// if we have exceeded the maximum size we can store in a single write, we reset and continue with a new one
		if(path_length >= 42)
		{
			//Write path_accum to 2D image
			write_imageui(ui4_dst_image, coords, ((union ul2_ui4)path_accum).ui4);
			// reset path_accum
			path_length = 0;
			path_accum = 0;
		}
	// 3 exit conditions, the current pixel is a start pixel, there is a significant change in arc angular rate, or the path accumulator is full
	// for now all 3 treated the same but technically the latter 2 should just reset part of the logic and continue the loop
	}while(!read_imagei(uc1_starts_image, coords).x);
}