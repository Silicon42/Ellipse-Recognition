#include "cast_helpers.cl"
#include "offsets_LUT.cl"
#include "math_helpers.cl"
#include "link_macros.cl"
// we define line segments as having midpoints that when doubled, don't differ from the endpoint by more than 1 pixel
//NOTE: all memory accesses to the 2D texture are basically random on a work item level and have minimal 2D locality within the
// a single work item due to segments traversing the image and the majority may likely be cache misses, so they are kept to an
// absolute minimum,
//TODO: verify if image type may not be ideal type for arguments

kernel void line_segments(
	read_only image2d_t uc1_cont_info,
	read_only image1d_t is2_start_coords,
	write_only image2d_t ic2_line_data,
	write_only image1d_t us1_line_counts)
{
	short index = get_global_id(0);	// must be scheduled as 1D
	
	// initialize variables of line segment tracing loop for first iteration
	int2 coords = read_imagei(is2_start_coords, index).lo;	// current pixel coordinates
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
		//TODO: once the duplicate processing bugs are fixed, remove this (currently fixed but future changes might break again)
		/*if(seg_count > 255)
		{
			printf("seg_count over\n");
			break;
		}*/
		for(int len = 1; ; ++len) //real base case exit condition is mid-block at len >= 127
		{
			cont_data = read_imageui(uc1_cont_info, coords).x;

			cont_idx = cont_data & R_CONT_IDX_MASK;
			offset_end += offsets_c[cont_idx];

			//check that data wasn't a start OR an end was signalled last pixel
			to_end |= cont_data & IS_START;//(cont_data & (IS_START | HAS_R_CONT)) != HAS_R_CONT;
			if(to_end)
				break;
			to_end = cont_data & IS_END_ADJ;

			coords += offsets[cont_idx];
			offset_x2_mid += offsets_c[path_hist[(len >> 1) & 0x1F]];
			// if 2* the midpoint is further than 1 pixel from the endpoint (2px^2 == 4) OR length exceed maximum allowed
			//count of applied offsets is 1 higher than len so need to exit at 126 with changes below
			if(len > 126)
				break;
			if(mag2_2d_c(offset_end - offset_x2_mid) > 4)
			{	//FIXME: This is a temporary fix to better smooth the segment transitions,
				// a proper fix would involve only writing out the midpoint segment,
				// and recycling the remaining half of the offsets to continue lengthening the newly halved line without breaking
			//	printf("%i	%i,	%i	%i\n", offset_x2_mid.x, offset_x2_mid.y, offset_x2_mid.x >> 1, offset_x2_mid.y >> 1);
				offset_x2_mid >>= 1;
				int2 offset_mid = convert_int2(offset_x2_mid);
				if(!(offset_mid.x || offset_mid.y))	// not sure this is actually possible but it doesn't hurt for now
				{
					printf(" midpoint 0 ");
					break;
				}
				++seg_count;
				//printf("%i %i \n", base_coords.x, base_coords.y);
				write_imagei(ic2_line_data, base_coords, (int4)(offset_mid, 0, -1));
				base_coords += offset_mid;
				offset_end -= offset_x2_mid;
				break;
			}

			//addition to offset_mid delayed to keep narrower distance threshold range, may or may not be ideal solution
			if(len < 64)
				path_hist[len & 0x1F] = cont_idx;
		}
		// wind back 1 pixel to last position where it was any of the following 
		// depending on which exit condition occured:
		// 2*midpoint within 1 pixel of endpoint / within max length / didn't overrun a start or end
		offset_end -= offsets_c[cont_idx];
		if(offset_end.x || offset_end.y)	//FIXME: this check shouldn't be neccessary
			write_imagei(ic2_line_data, base_coords, (int4)(convert_int2(offset_end), 0, -1));
		else
		{
		//	printf("%i %i \n", base_coords.x, base_coords.y);
			--seg_count;
		}

	} while(!to_end);
	
	//printf("%i\n", index);

	write_imageui(us1_line_counts, index, seg_count);
}