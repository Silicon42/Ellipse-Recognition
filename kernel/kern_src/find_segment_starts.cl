// Kernel meant to select intial line/arc segment starting points in a non-max
// suppressed edge image (such as after Canny) 

// VVVVV not implemented yet, using a serial reduction to 1D as a separate kernel for now
//and hash them into a 1D array.
// Since starts should be extremely sparse, a sufficiently large hash table should
// have few collisions but still take up less space and have better access patterns
// than operating on the whole image. If it can't be made sufficiently big enough,
// then the secondary output can be used to confirm remaining starts after transformation
// back into uncompressed space and processing on those remaining can be doen in a
// 2nd(or more) pass

//FIXME: it seems there is some rare corner case where an edge segment won't have a start, revisit this when I have more insight
#include "cast_helpers.cl_h"
#include "offsets_LUT.cl_h"
#include "link_macros.cl_h"
//FIXME: replace temp fix for multiple definition by adding proper support for included sources
//constant const int2 offsets[] = {(int2)(1,0),1,(int2)(0,1),(int2)(-1,1),(int2)(-1,0),-1,(int2)(0,-1),(int2)(1,-1)};

//NOTE: returned values are in the form 0bS000Errr where 
// "S" is the start indicator flag,
// "E" is not end adjacent indicator flag
//   ie. next pixel cont_data not valid to read as part of current chain,
//   handles Y-junctions and ends of chain, doesn't handle start being next
// "r" is the 3-bit direction index

//TODO: need to add an is_supported flag so that small segments that support other separately detected small segments don't get deleted
// This might be decently involved to actually implement
kernel void find_segment_starts(
	read_only image2d_t ic1_grad_ang,
	read_only image2d_t uc1_cont,
	write_only image2d_t uc1_starts_cont)
{
	const int2 coords = (int2)(get_global_id(0), get_global_id(1));

	uchar cont_data = read_imageui(uc1_cont, coords).x;

	// if no valid right continuation, cannot be start or have valid cont data, so vast majority returns early
	if(!(cont_data & HAS_R_CONT))
		return;
	
	// else there is a valid right continuation

	// read next pixel in the chain (right continuation) to verify this is a true/mutual connection
	uchar r_cont_idx = cont_data & R_CONT_IDX_MASK;
	int2 adjacent_coords = coords + offsets[r_cont_idx];
	uchar adjacent_data = read_imageui(uc1_cont, adjacent_coords).x;
	// y-junction prevention, stops multiple edges that would join to process a shared region
	// also detects if next pixel is a normal end pixel, a pixel is end adjacent if either:
	// 1) the right continuation's left continuation is not mutual,
	//     i.e. a joining y-junction where the current pixel is not part of the through connection,
	// 2) or the next pixel in the chain (right continuation) has no right continuation itself
	uchar is_r_mutual = ((adjacent_data >> L_CONT_IDX_SHIFT) ^ r_cont_idx) == 0b1100;
	uchar isnt_end_adjacent = is_r_mutual ? (adjacent_data & HAS_R_CONT) : 0;

	// end adjacent pixels aren't allowed to be starts, this discards single and 2 pixel edge chains from being processed
	// since they also don't have valid continuation data, nothing needs to be written for them
	if(!isnt_end_adjacent)
		return;

	uchar out_data = r_cont_idx | isnt_end_adjacent;

	// if a pixel has both continuations it can only become a start if it qualifies as a potential loop-breaking start
	if(cont_data & HAS_L_CONT)
	{
		// to qualify for a loop breaking start, the grad angle must be non-negative...
		char grad_ang = read_imagei(ic1_grad_ang, coords).x;
		if(grad_ang >= 0)
		{
			// and the gradient angle of the right neighbor must be negative
			grad_ang = read_imagei(ic1_grad_ang, adjacent_coords).x;
			out_data |= (grad_ang < 0) ? IS_START : 0;
		}
	}
	else
		out_data |= IS_START;

	write_imageui(uc1_starts_cont, coords, out_data);
}