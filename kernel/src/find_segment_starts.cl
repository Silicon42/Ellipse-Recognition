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
#include "cast_helpers.cl"
#include "neighbor_utils.cl"
#include "offsets_LUT.cl"
#include "link_macros.cl"

//NOTE: returned values are in the form 0bSE0lriii where 
// "S" is the start indicator flag,
//TODO: v swap this flag's sense so that overruns into unwritten pixels aren't possible
// "E" is end adjacent indicator flag, ie. cont_data still valid but next pixel coord won't be, doesn't handle start being next
// "l" is the left support indicator flag,
// "r" is occupancy/right continuation indicator flag, and
// "i" is the 3-bit direction index

//TODO: need to add an is_supported flag so that small segments that support other separately detected small segments don't get deleted
// This might be decently involved to actually implement
kernel void find_segment_starts(
	read_only image2d_t uc1_cont,
	read_only image2d_t iC1_grad_ang,
	write_only image2d_t uc1_starts_cont)
{
	const int2 coords = (int2)(get_global_id(0), get_global_id(1));

	uchar cont_data = read_imageui(uc1_cont, coords).x;

	char grad_ang;
	uchar adjacent_data, adjacent_idx;
	int2 adjacent_coords;
	uchar is_end_adjacent = 0;	//also used for early rejection of unconnected 2-pixel segments

	// y-junction prevention, stops multiple edges that would join to process a shared region
	if(cont_data & HAS_R_CONT)	// if valid right continuation
	{
		adjacent_idx = cont_data & R_CONT_IDX_MASK;
		adjacent_coords = coords + offsets[adjacent_idx];
		adjacent_data = read_imageui(uc1_cont, adjacent_coords).x;
		// right continuation's left continuation is not mutual,
		// i.e. a joining y-junction where the current pixel is not part of the through connection,
		// then set is_end_adjacent flag to force an edge processing stop
		//NOTE: right continuation's left continuation is implicitly populated by fact that this cell exists
		is_end_adjacent = !(adjacent_data & HAS_R_CONT) || ((adjacent_data >> L_CONT_IDX_SHIFT) ^ adjacent_idx) != 4;
}

	switch(cont_data & HAS_BOTH_CONT)
	{
	default:	// if cont_data has no left or right continuation flag, it was 
				// not an edge and therefore has no continuation flags set,
		return;	// therefore this work item can exit early, vast majority exits here
		//TODO: might have better perf if this is moved earlier as a separate if
		/*	// or if it would be a standard right end, there is no need to keep the pixel so it can also be handled by the default case
	case HAS_LEFT_CONT:
		cont_data &= HAS_LEFT_CONT;
		break;*/
	case HAS_BOTH_CONT:	// both sides have a continuation
		adjacent_idx = cont_data >> L_CONT_IDX_SHIFT;
		cont_data &= 0x1F;	// only right continuation and left support flag will ever be written to output regardless of path taken from this point
		adjacent_coords = coords + offsets[adjacent_idx];
		adjacent_data = read_imageui(uc1_cont, adjacent_coords).x & 0xF;
		// if the left continuation is a mutual link, it is likely not a start but needs more logic.
		// the one exception is if it qualifies for a loop breaking start,
		// else if it's not a mutual link, there was a fork and this is a start
		if((adjacent_data ^ adjacent_idx) == 0xC)
		{
			grad_ang = read_imagei(iC1_grad_ang, coords).x;
			if(grad_ang < 0)	// to qualify for a loop breaking start, the grad angle must be non-negative
				break;	//pass on only continuation data, no start flag

			grad_ang = read_imagei(iC1_grad_ang, adjacent_coords).x;
			if(grad_ang > 0)	// the gradient angle of the left neighbor must be negative
				break;	//pass on only continuation data, no start flag
		}
		// fall-through to add start flag
	case HAS_R_CONT:
		// if it starts and ends on the same pixel or an adjacent pixel,
		// it's not usable data and shouldn't be marked as a start
		if(is_end_adjacent)
			break;
		
		cont_data |= IS_START;
	}

	write_imageui(uc1_starts_cont, coords, cont_data | (is_end_adjacent << END_ADJ_SHIFT));
}