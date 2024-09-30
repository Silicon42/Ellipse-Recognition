#include "cast_helpers.cl"
#include "offsets_LUT.cl"
//#include "line_data.cl"
#include "math_helpers.cl"
#include "link_macros.cl"
// we define line segments as having midpoints that when doubled, don't differ from the endpoint by more than 1 pixel
//NOTE: all memory accesses to the 2D texture are basically random on a work item level and have minimal 2D locality within the
// a single work item due to segments traversing the image and the majority may likely be cache misses, so they are kept to an
// absolute minimum,
//TODO: verify if image type may not be ideal type for arguments

kernel void line_segments(
	read_only image1d_t iS2_start_coords,
	read_only image2d_t uc1_cont_info,
	write_only image1d_t us1_line_counts,
	write_only image2d_t iC2_line_data)
{
	short index = get_global_id(0);	// must be scheduled as 1D
	
	// initialize variables of line segment tracing loop for first iteration
	int2 coords = read_imagei(iS2_start_coords, index).lo;	// current pixel coordinates
	if(!((union l_conv)coords).l)	// this does mean a start at (0,0) won't get processed but I don't think that's particularly likely to happen and be critical
		return;
	
	uchar cont_data, cont_idx, /*is_supported,*/ to_end = 0;
	cont_data = read_imageui(uc1_cont_info, coords).x;
	//is_supported = cont_data & HAS_L_CONT;	// there was another supporting pixel at the start
	//NOTE: ^ this is mostly for the sometimes short segments that loop start injection can cause so they don't get thrown out
	
	// start specified in start_info implicitly has a valid continuation, so can be safely masked to just index
	cont_idx = cont_data & R_CONT_IDX_MASK;

	// ring buffer that stores history of pixels traversed,
	// is 1/4 size because only half the length must be recorded for finding the midpoint
	// and half of that is already accumulated in offset_x2_mid at any given time
	uchar path_hist[32];
	//struct line_AB_tracking lines = {0};
	char2 offset_x2_mid, offset_end = 0;
	int2 base_coords = coords;
	ushort seg_count = 0;
	coords += offsets[cont_idx];

	// exit condition occurs when a read pixel indicates a start or 1 pixel after a pixel indicates it's end adjacent
	do
	{
		base_coords += convert_int2(offset_end);
		offset_x2_mid = offset_end = offsets_c[cont_idx];
		path_hist[0] = cont_idx;
		++seg_count;

		for(int len = 1; ; ++len) //real base case exit condition is mid-block at len >= 127
		{
			cont_data = read_imageui(uc1_cont_info, coords).x;

			cont_idx = cont_data & R_CONT_IDX_MASK;
			offset_end += offsets_c[cont_idx];

			//check that data wasn't a start OR an end was signalled last pixel
			to_end = cont_data & IS_START;
			if(to_end)
				break;
			to_end = cont_data & IS_END_ADJ;

			coords += offsets[cont_idx];
			// if 2* the midpoint is further than 1 pixel from the endpoint OR length exceed maximum allowed
			if(mag2_2d_c(offset_end - offset_x2_mid) > 2 || len >= 127)
				break;

			//addition to offset_mid delayed to keep narrower distance threshold range, may or may not be ideal solution
			offset_x2_mid += offsets_c[path_hist[(len >> 1) & 0x1F]];
			if(len < 64)
				path_hist[len & 0x1F] = cont_idx;
		}
		// wind back 1 pixel to last position where it was any of the following 
		// depending on which exit condition occured:
		// 2*midpoint within 1 pixel of endpoint / within max length / didn't overrun a start or end
		offset_end -= offsets_c[cont_idx];
		write_imagei(iC2_line_data, base_coords, (int4)(convert_int2(offset_end), 0, -1));
	} while(!to_end);

	write_imageui(us1_line_counts, index, seg_count);
}