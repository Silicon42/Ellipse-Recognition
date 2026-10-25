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

	// buffer that stores history of directions traversed,
	uchar path_hist[127];
	int2 x2_mid_coords;
	int2 base_coords;
	ushort seg_count = 0;

	// exit condition occurs when a read pixel indicates a start or 1 pixel after a pixel indicates it's end adjacent
	do	// while(!to_end)
	{
		base_coords = x2_mid_coords = coords;
		// error message, this should never happen
		if(any(base_coords < 0 || base_coords >= bounds))
			printf("OOPS1: (%i, %i)\n", base_coords.x, base_coords.y);

	//	if(!index)
	//		printf("\nS(%i,%i) ", base_coords.x, base_coords.y);
		//TODO: once the duplicate processing bugs are fixed, remove this (currently fixed but future changes might break again)
		/*if(seg_count > 255)
		{
			printf("seg_count over\n");
			break;
		}*/
		char len = 0;
		while(1)	//real base case exit condition is mid-block at len  127
		{
			cont_data = read_imageui(uc1_cont_info, coords).x;

			// check that current pixel isn't a start to prevent double processing,
			// if it is, it must immediately exit without applying the current pixel's offset to coords
			// uses xor so that the initial start unsets the flag to_end and any further starts set it again
			to_end ^= cont_data & IS_START;
			// check that current pixel's next index data will be valid
			// if it isn't, it must exit after applying the current pixel's offset
			to_end |= !(cont_data & ISNT_END_ADJ);
			if(to_end)
				break;

			cont_idx = cont_data & R_CONT_IDX_MASK;
			coords += offsets[cont_idx];
		//	if(!index)
		//		printf("%i (%i, %i) ", cont_idx, coords.x, coords.y);

			path_hist[len] = cont_idx;

			x2_mid_coords += offsets[path_hist[(len/2)]];
			// count of applied offsets must never exceed 127 to prevent potential over/underflow
			// len is 1 less than the applied offsets at this point as it's used to index into the path_hist array
			if(len == 126)
				break;

			// if 2* the midpoint is further than 2 pixel taxicab distance from the endpoint
			if(taxi_len_2d_i(coords - x2_mid_coords) > 2)
			{
				// traverse the list in reverse to counter late corner detection bias
				// when we encounter the exit condition, that means the midpoint of the reverse traversal was the real end
				char rev_len = 0;
				int2 rev_end_offset;
				rev_end_offset = x2_mid_coords = 0;
				
				while(rev_len < len)
				{
					rev_end_offset += offsets[path_hist[len - rev_len]];
					x2_mid_coords += offsets[path_hist[len - rev_len/2]];
					++rev_len;
					if(taxi_len_2d_i(rev_end_offset - x2_mid_coords) > 2)
					{
						if(rev_len & 1)	//if rev_len is odd, remove the odd entry
							x2_mid_coords += offsets[path_hist[len - rev_len/2]];
						
						coords -= x2_mid_coords/2;
					//TODO: this throws away some of the work in the form of the reads that were done past the above mentioned point. 
					// This could be prevented with some additional logic for scavenging them but the changes aren't trivial and might
					// be slower than just re-reading them since they are almost certainly in the cache so this should be checked.
						break;
					}
				}
				break;
			}
			++len;
		}

		if(len)
		{
			// error message, this should never happen
			if(all(coords == base_coords))
				printf("0 offset: (%i, %i) %i\n", base_coords.x, base_coords.y, len);

			++seg_count;
			write_imagei(ic2_line_data, base_coords, (int4)(coords - base_coords, 0, -1));
		}
	} while(!to_end);
	
	//printf("%i\n", index);

	write_imageui(us1_line_counts, index, seg_count);
}