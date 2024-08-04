#include "cast_helpers.cl"
#include "path_struct_defs.cl"
#define ACCEL_THRESH 20
//NOTE: all memory accesses to the 2D texture are basically random on a work item level and have minimal 2D locality within the
// a single work item due to segments traversing the image and the majority may likely be cache misses, so they are kept to an
// absolute minimum

kernel void arc_segments(read_only image1d_t us2_start_info, read_only image2d_t uc1_cont_image, read_only image2d_t iC1_grad_image, write_only image2d_t ui4_path_image, write_only image2d_t uc1_trace)
{
	union {
		long16 combined;
		int2 coords[16];
	} xy_hist;
	xy_hist.combined = -1;
	union l_conv bounds, coords, base_coords, prev_coords;
	bounds.i = get_image_dim(ui4_path_image);
	const int2 offsets[] = {(int2)(0,1),(int2)(-1,1),(int2)(-1,0),-1,(int2)(0,-1),(int2)(1,-1),(int2)(1,0),1,
							(int2)(0,1),(int2)(-1,1),(int2)(-1,0),-1,(int2)(0,-1)};	// repeat for addition overrun
	short index = get_global_id(0);	// must be scheduled as 1D

	// initialize variables of arcs segment tracing loop for first iteration
	coords.ui = read_imageui(us2_start_info, index).lo;
	if(!coords.l)	// this does mean a start at (0,0) won't get processed but I don't think that's particularly likely to happen and be critical
		return;
	
	base_coords = coords;

	// the cached 0-deg-relative bin index for where the next pixel was found when finding segment starts
	uchar cont_idx = read_imageui(uc1_cont_image, coords.i).x & 7;	//	start specified in start_info is assumed to be valid and have start flag, so can be safely masked to just index
	uchar cont_data = 0;
	write_imageui(uc1_trace, coords.i, -1);

	char prev_angle, curr_angle, prev_angle_diff, curr_angle_diff, angle_accel, max_accel, min_accel;
	prev_angle = read_imagei(iC1_grad_image, coords.i).x;
	coords.i += offsets[cont_idx];
//	if(any(coords.u >= bounds.u))
//		return;
	write_imageui(uc1_trace, coords.i, -1);

	curr_angle = read_imagei(iC1_grad_image, coords.i).x;

	curr_angle_diff = curr_angle - prev_angle;

	// we define arc segments as having a roughly constant angular rate of change along their path_length within some minimum noise floor,
	// this means that we want to track overall drift of this angular "acceleration" which we would like to stay near 0 and if
	// it deviates, then we need to start a new arc segment and reset the angular acceleration to be just based on the last 3 pixels?? might be slightly off with that logic
	angle_accel = max_accel = min_accel = 0;
	ulong2 path_accum = (ulong2)(cont_idx,0);	//shift register that stores return data for what pixels were included in the calculation
	uchar path_length = 1;
	cont_data = read_imageui(uc1_cont_image, coords.i).x;

	ushort watchdog = 0;

	// loop until we hit a start or run out of pixels for the arc segment
	while((cont_data & 0x88) == 0x80)	// check valid and not start
	{
		++watchdog;
		// infinite loop prevention
		//FIXME: this is a temporary fix, a proper logical fix that prevents the loops from forming needs to be added
		if(!watchdog)
		{
			printf("watchdog: segment too long. Is it looped?: (%i, %i), curr_angle: %i, prev_angle: %i\n", coords.i.x, coords.i.y, curr_angle, prev_angle);
			break;
		}
		
		cont_idx = cont_data & 7;	// clear validity to use as index
		prev_coords = coords;
		xy_hist.coords[watchdog & 0xF] = coords.i;
		coords.i += offsets[cont_idx];
		if(any(xy_hist.combined == coords.l))
		{
			printf("Loop @ (%i, %i)\n", coords.i);
			break;
		}
		//valid location check, if we ever go off the edges of the image there is no continuation possible
		if(any(coords.ui >= bounds.ui))
		{
			printf("line ended out of bounds\n");
			break;
		}
		// update angle tracking variables
		prev_angle = curr_angle;
		// read angle + continuation data from inputs for this pixel
		curr_angle = read_imagei(iC1_grad_image, coords.i).x;
		cont_data = read_imageui(uc1_cont_image, coords.i).x;

		// update angle diff tracking variables
		prev_angle_diff = curr_angle_diff;
		curr_angle_diff = curr_angle - prev_angle;
		if(cont_idx & 1)	// if it's a diagonal the diff is less due to being spread over a greater distance
			curr_angle_diff *= M_SQRT1_2_F;
		
		// update angle accel tracking variables
		angle_accel += curr_angle_diff - prev_angle_diff;
		max_accel = angle_accel > max_accel ? angle_accel : max_accel;
		min_accel = angle_accel < min_accel ? angle_accel : min_accel;

		// if the total angular acceleration for the arc exceeds the small acceleration threshold,
		// save the accumulator and restart it as a new arc segment
		if(max_accel - min_accel > ACCEL_THRESH)	// this corresponds to a ??? deg per pixel^2 total acceleration noise threshold
		{
			angle_accel = max_accel = min_accel = 0;
			//Write path_accum to 2D image
			write_data_accum(path_accum, path_length, ui4_path_image, base_coords.i);
			// reset path_accum
			base_coords = prev_coords;
			path_length = 1;
			path_accum = (ulong2)(cont_idx, 0);
		}
		else
		{
			//if we've filled a 64-bit value copy it to the next one and continue
			switch(path_length)
			{
			// if we have exceeded the maximum size we can store in a single write, we reset and continue with a new one
			case ACCUM_STRUCT_LEN2:
				//Write path_accum to 2D image
				write_data_accum(path_accum, LEN_BITS_MASK, ui4_path_image, base_coords.i);
				// reset path_accum
				base_coords = prev_coords;
				path_length = 1;
				path_accum = (ulong2)(cont_idx, 0);
				break;
			// if we have exceeded the maximum size we can store in a single long, swap to the 2nd long in the accumulator
			case ACCUM_STRUCT_LEN1:
				path_accum = path_accum.yx;
			default:
				path_accum.x |= (ulong)cont_idx << (3 * (path_length - (path_length < ACCUM_STRUCT_LEN1 ? 0 : ACCUM_STRUCT_LEN1)));
				++path_length;
			}
		}
		// mark current pixel in debug trace
		write_imageui(uc1_trace, coords.i, -1);
	}

	// if edge was a lone edge of no more than 6 pixels, reject as noise
//	printf("len: %i	", watchdog);
	if(watchdog < 5 && (cont_data & 0x88) != 0x88)
		return;

	write_data_accum(path_accum, path_length, ui4_path_image, base_coords.i);
}