//#include "cast_helpers.cl_h"
#include "offsets_LUT.cl_h"
#include "math_helpers.cl_h"
#include "link_macros.cl_h"
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
	int2 bounds = get_image_dim(uc1_cont_info);	//DEBUGGING CODE remove once certain no more issues remain
	
	// initialize variables of line segment tracing loop for first iteration
	int2 coords = read_imagei(is2_start_coords, index).lo;	// current pixel coordinates
	if(all(coords == 0))	// this does mean a start at (0,0) won't get processed but I don't think that's particularly likely to happen and be critical
		return;

//	printf("%v2i\n", coords);
	
	uchar cont_data, cont_idx, to_end = IS_START;

	// ring buffer that stores history of pixels traversed,
	// is 1/4 size because only half the length must be recorded for finding the midpoint
	// and half of that is already accumulated in offset_x2_mid at any given time
	uchar path_hist[32];
	int2 offset_x2_mid, offset_end;
	int2 base_coords;
	ushort seg_count = 0;

	// exit condition occurs when a read pixel indicates a start or 1 pixel after a pixel indicates it's end adjacent
	do	// while(!to_end)
	{
		base_coords = coords;
	//	if(!index)
	//		printf("\nS(%i,%i) ", base_coords.x, base_coords.y);
		offset_x2_mid = offset_end = 0;

		//TODO: once the duplicate processing bugs are fixed, remove this (currently fixed but future changes might break again)
		/*if(seg_count > 255)
		{
			printf("seg_count over\n");
			break;
		}*/
		int len = 0;
		while(1)	//real base case exit condition is mid-block at len >= 125
		{
			cont_data = read_imageui(uc1_cont_info, coords).x;

			// check that current pixel isn't a start to prevent double processing,
			// if it is, it must immediately exit without applying the current pixel's offset to offset_end
			// uses xor so that the initial start unsets the flag to_end and any further starts set it again
			to_end ^= cont_data & IS_START;
			// check that current pixel's next index data will be valid
			// if it isn't, it must exit after applying the current pixel's offset
			to_end |= !(cont_data & ISNT_END_ADJ);
			if(to_end)
				break;

			cont_idx = cont_data & R_CONT_IDX_MASK;
			int2 prev_offset = offset_end;
			offset_end += offsets[cont_idx];
//			if(all(offset_end == 0))
//				printf("%v2i offset: (%i, %i) %i %i\n", coords, base_coords.x, base_coords.y, (int)cont_idx, seg_count);

			coords += offsets[cont_idx];
		//	if(!index)
		//		printf("%i (%i, %i) ", cont_idx, coords.x, coords.y);

			if(len < 63)
				path_hist[len & 0x1F] = cont_idx;

			offset_x2_mid += offsets[path_hist[(len/2) & 0x1F]];
			// if 2* the midpoint is further than 2 pixel taxicab distance from the endpoint OR length exceed maximum allowed
			// count of applied offsets is 1 higher than len so need to exit at 126 with changes below
			if(len >= 125)
				break;

			// if there are at least 2 pixels
			if(taxi_len_2d_i(offset_end - offset_x2_mid) > 2)
			{			
			//FIXME: The below block was disabled because it led to too many situations
			// where multiple points could be in a line and cause degenerate conics to be calculated
			// this might be fixed by detecting those situations and joining the straight segments
			//FIXME: This is a temporary fix to better smooth the segment transitions,
			// a proper fix would involve only writing out the midpoint segment,
			// and recycling the remaining half of the offsets to continue lengthening the newly halved line without breaking
			//	printf("offset: <%i, %i> 2*mid: <%i, %i> ", offset_end.x, offset_end.y, offset_x2_mid.x, offset_x2_mid.y);
				offset_x2_mid /= 2;
				if(!(offset_x2_mid.x || offset_x2_mid.y))	// not sure this is actually possible but it doesn't hurt for now
				{
					printf(" midpoint 0 ");
					break;
				}
				++seg_count;
				//printf("%i %i \n", base_coords.x, base_coords.y);
				if(any(base_coords < 0 || base_coords >= bounds))
					printf("OOPS1: (%i, %i)\n", base_coords.x, base_coords.y);
				write_imagei(ic2_line_data, base_coords, (int4)(offset_x2_mid, 0, -1));
				base_coords += offset_x2_mid;
				offset_end -= offset_x2_mid;
				break;
			}
			++len;
		}

		if(len)
		{
			// error messages, these should never happen
			if(all(offset_end == 0))
				printf("0 offset: (%i, %i) %i\n", base_coords.x, base_coords.y, len);
			if(any(base_coords < 0 || base_coords >= bounds))
				printf("OOPS2: (%i, %i)\n", base_coords.x, base_coords.y);

			++seg_count;
			write_imagei(ic2_line_data, base_coords, (int4)(offset_end, 0, -1));
		}
	} while(!to_end);
	
	//printf("%i\n", index);

	write_imageui(us1_line_counts, index, seg_count);
}