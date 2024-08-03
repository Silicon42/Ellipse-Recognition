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
#include "samplers.cl"
//NOTE: return value is in the form 0bv###siii where "i" is the 3-bit direction index, "s" denotes if the entry is a start,
// "v" denotes if the index is valid (may be invalid due to multiple nearby continuations), and # is unassigned padding/occupancy indicator

kernel void find_segment_starts(read_only image2d_t iC1_edge_image, write_only image2d_t uc1_starts_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	char grad_ang = read_imagei(iC1_edge_image, coords).x;

	// magnitude channel used as bool for occupancy
	// if gradient angle == 0, it wasn't set in canny_short because even 0 should have the occupancy flag set,
	// therefore this work item isn't on an edge and can exit early, vast majority exits here
	if(!grad_ang)
		return;

	// read adjacent pixels into an array such that elements progress counter-clockwise around the central pixel
	// starting from the 9 o'clock position, this is so that a left rotate() by a proportional amount to the angle
	// of the normal partitions the pixels such that all pixel values to the right of the normal are in the lower half
	// and values to the left are in the upper half
	union l_conv neighbors;
	neighbors.c.s0 = read_imagei(iC1_edge_image, clamped, coords + (int2)(-1, 0)).x;
	neighbors.c.s1 = read_imagei(iC1_edge_image, clamped, coords + (int2)(-1, 1)).x;
	neighbors.c.s2 = read_imagei(iC1_edge_image, clamped, coords + (int2)( 0, 1)).x;
	neighbors.c.s3 = read_imagei(iC1_edge_image, clamped, coords + 1).x;
	neighbors.c.s4 = read_imagei(iC1_edge_image, clamped, coords + (int2)( 1, 0)).x;
	neighbors.c.s5 = read_imagei(iC1_edge_image, clamped, coords + (int2)( 1,-1)).x;
	neighbors.c.s6 = read_imagei(iC1_edge_image, clamped, coords + (int2)( 0,-1)).x;
	neighbors.c.s7 = read_imagei(iC1_edge_image, clamped, coords - 1).x;

	// convert gradient angle to index of bin corresponding to it
	uchar grad_idx = (uchar)grad_ang >> 5;
	// left rotate the neighbors so that indexing is now relative to the gradient vector
	// rotation amount is grad_idx * 8 bits per byte
	neighbors.l = rotate(neighbors.l, 8L * grad_idx);

	long occupancy = neighbors.l & 0x0101010101010101;
	occupancy = (occupancy << 8) - occupancy;
	union l_conv diff, is_diff_small;
	diff.uc = abs(neighbors.c - grad_ang);
	is_diff_small.c = diff.uc < (uchar)32;
	is_diff_small.l &= occupancy;
	uchar rel_idx;
	// exactly one neighbor on this side must be populated for this to potentially count as a start
	switch(is_diff_small.i.lo)
	{
	default:	// indeterminate, multiple adjacent pixels have possibly valid continuations
		//TODO: it might be useful to OR in the occupancy of the nearby pixels in place of the start flag and index in this situation
		printf("how did you get here?\n");
		//write_imageui(uc1_starts_image, coords, 0x70);	//	indicate occupied but indeterminate pixel
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
	switch(is_diff_small.i.hi)
	{
		default:	// or in the special corner case of a closed loop, we add a start at the zero crossing
			if(grad_ang >= 32 || grad_ang < -32)
				break;
			roll_over.c = grad_ang ^ neighbors.c.hi;
			roll_over.i &= 0xE0E0E0E0 & is_diff_small.i.hi;
			if(all(roll_over.c != (char)0xE0))
				break;
			// fall-through
		case 0:	// all left side neighbors empty or dissimilar
			out_data |= 8;	// set start flag
	}

	write_imageui(uc1_starts_image, coords, out_data);
}