//stripped down version of arc builder that doesn't do any ellipse calculation, just the logical checks
#include "cast_helpers.cl"
#include "math_helpers.cl"

kernel void arc_builder_stripped(
	read_only image1d_t is2_start_coords,
	read_only image1d_t us1_line_counts,
	read_only image2d_t ic2_line_data,
	write_only image2d_t us1_seg_in_arc)
{
	short index = get_global_id(0);	// must be scheduled as 1D
	if(!index)
		printf("arc_builder\n");
	int2 base_coords = read_imagei(is2_start_coords, index).lo;	// current pixel coordinates
	if(!((union l_conv)base_coords).l)	// this does mean a start at (0,0) won't get processed but I don't think that's particularly likely to happen and be critical
		return;

	int remaining_segs = read_imageui(us1_line_counts, index).x;
	if(!remaining_segs)
		return;
		
	int2 total_offset, curr_seg, prev_seg;
	curr_seg = read_imagei(ic2_line_data, base_coords).lo;
	total_offset = 0;
	char reset = 0;
	ushort seg_cnt = 1;
	char dir, dir_trend = 0;
	int dir_cross;
	uchar kick = 0;

	// loop over all segments that came from this start
	// don't have to worry about returning to start b/c with the forward acute angle restriction
	// that would require at least 5 segments and therefore wouldn't end up with one of the points
	// as (0,0) on the initial calculation
	while(--remaining_segs)
	{
		if(reset)
		{
			reset = 0;
			write_imageui(us1_seg_in_arc, base_coords, seg_cnt);

			base_coords += total_offset;
			//if(remaining_segs < 4)
			//	write_imageui(us1_seg_in_arc, base_coords, seg_cnt);
			total_offset = 0;	//keep last segment that caused the reset
			seg_cnt = 1;
			dir_trend = 0;	//trend unknown since only 1 segment
		}
		prev_seg = curr_seg;
		total_offset += curr_seg;
		curr_seg = read_imagei(ic2_line_data, base_coords + total_offset).lo;

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
		
		dir = (dir_cross < 0) ? -1 : dir_cross > 0;	//extract sign of dir_cross to get just the curving direction
		// if curving direction changes between +/- trigger a reset
		if((dir ^ dir_trend) == -2)
		{
			reset = 1;
			continue;
		}
		// if curving direction hasn't yet collapsed to +/-1, attempt to do so
		if(!dir_trend)
			dir_trend = dir;
		

		// this must stay at the end b/c some situations need to be able to skip it
		++seg_cnt;
	}
	//flush last arc
	write_imageui(us1_seg_in_arc, base_coords, seg_cnt);
}
