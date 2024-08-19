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
//NOTE: return value is in the form 0bS00soiii where "S" is the start indicator flag, "s" is the support indicator flag
// "o" is occupancy indicator flag, "i" is the 3-bit direction index,

//TODO: need to add an is_supported flag so that small segments that support other separately detected small segments don't get deleted
// This might be decently involved to actually implement
kernel void find_segment_starts(read_only image2d_t uc1_cont, read_only image2d_t iC1_grad_ang, write_only image2d_t uc1_starts_cont)
{
	const int2 coords = (int2)(get_global_id(0), get_global_id(1));
	const int2 offsets[] = {(int2)(1,0), 1, (int2)(0,1), (int2)(-1,1), (int2)(-1,0), -1, (int2)(0,-1), (int2)(1,-1)};

	uchar cont_data = read_imageui(uc1_cont, coords).x;

	uchar left_data, left_idx;
	char grad_ang;
	int2 left_coords;
	switch(cont_data & 0x18)
	{
	default:	// if cont_data is null, it was not an edge and therefore has no continuation flags set,
				// or if it had only the left continuation flag set it has nowhere to continue to
		return;	// therefore this work item can exit early, vast majority exits here

	case 0x18:	// both sides have a continuation
		left_idx = cont_data >> 5;
		cont_data &= 0x1F;	// only right continuation and left support flag will ever be written to output regardless of path taken from this point
		left_coords = coords + offsets[left_idx];
		left_data = read_imageui(uc1_cont, left_coords).x & 15;

		// if the left continuation is a mutual link, it is likely not a start but needs more logic.
		// the one exception is if it qualifies for a loop breaking start
		// else if it's not a mutual link, there was a fork and this is a start
		if((left_data ^ left_idx) == 0b1100)
		{
			grad_ang = read_imagei(iC1_grad_ang, coords).x;
			if(grad_ang < 0)	// to qualify for a loop breaking start, the grad angle must be non-negative
				break;	//pass on only continuation data, no start flag

			grad_ang |= read_imagei(iC1_grad_ang, left_coords).x;
			if((grad_ang & 0xC0) < 0)	// the gradient angle of the left neighbor must OR with it to a non-negative
				break;	//pass on only continuation data, no start flag
		}
	//	printf("!\n");
		// fall-through to add start flag
	case 0x08:
		cont_data |= 0xE0;
	}
/*
	if(!(cont_data & 16))
	{
		
	}
	// read adjacent pixels into an array such that elements progress counter-clockwise around the central pixel
	// starting from the 9 o'clock position, this is so that a left rotate() by a proportional amount to the angle
	// of the normal partitions the pixels such that all pixel values to the right of the normal are in the lower half
	// and values to the left are in the upper half
	union l_conv neighbors;
	neighbors.c = read_neighbors_ccw(iC1_edge_image, coords);

	// convert gradient angle to index of bin corresponding to it
	uchar grad_idx = (uchar)grad_ang >> 5;
	// left rotate the neighbors so that indexing is now relative to the gradient vector
	// rotation amount is grad_idx * 8 bits per byte
	neighbors.l = rotate(neighbors.l, 8L * grad_idx);

	long occupancy = get_occupancy_mask(neighbors.l);
	union l_conv diff, is_diff_small_mask;
	diff.c = neighbors.c - grad_ang;
	is_diff_small_mask.l = is_diff_small(diff.c, occupancy);
	uchar rel_idx;
	// exactly one neighbor on this side must be populated for this to potentially count as a start
	switch(is_diff_small_mask.i.hi)
	{
	default:	// indeterminate, multiple adjacent pixels have possibly valid continuations
		//TODO: it might be useful to OR in the occupancy of the nearby pixels in place of the start flag and index in this situation
		printf("how did you get here? (%i, %i) %X %x\n", coords, is_diff_small_mask.i);
		write_imageui(uc1_starts_cont, coords, 0x70);	//	indicate occupied but indeterminate pixel
	case 0:		// no similar angle continuations
		return;
	case 0x000000FF:
		rel_idx = 2;
		break;
	case 0x0000FF00:
		rel_idx = 1;
		break;
	case 0x00FF0000:
		rel_idx = 0;
		break;
	case 0xFF000000:
		rel_idx = -1;
	}
	
	uchar out_data = ((rel_idx + grad_idx) & 7) | 0xF0;	// convert from normal-relative index to 0 deg-relative index and OR occupancy flag/padding
	// all neighbors on this side must not be similar angles for this to count as a start
	union i_conv roll_over;
	switch(is_diff_small_mask.i.lo)
	{
		default:	// or in the special corner case of a closed loop, we add a start at the zero crossing
			if(grad_ang >= 32 || grad_ang < -32)
				break;
			roll_over.c = grad_ang ^ neighbors.c.hi;
			roll_over.i &= 0xE0E0E0E0 & is_diff_small_mask.i.lo;
			if(all(roll_over.c != (char)0xE0))
				break;
			// fall-through
		case 0:	// all left side neighbors empty or dissimilar
			out_data |= 8;	// set start flag
	}
*/
	write_imageui(uc1_starts_cont, coords, cont_data);
}