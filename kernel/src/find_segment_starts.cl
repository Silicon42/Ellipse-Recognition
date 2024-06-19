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

/*
union l_c8{
	long l;
	char8 c;
	char a[8];
};
*/
union i_c4{
	int i;
	char4 c;
	uchar4 uc;
	uchar uca[4];
};

kernel void find_segment_starts(read_only image2d_t iC1_edge_image, write_only image2d_t uc1_starts_image)
{
	const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	char grad_ang = read_imagei(iC1_edge_image, coords).x;

	// magnitude channel used as bool for occupancy
	// if gradient angle == 0, it wasn't set in canny_short because even 0 should have the occupancy flag set,
	// therefore this work item isn't on an edge and can exit early, vast majority exits here
	if(!grad_ang)
		return;

	// read adjacent pixels into an array such that elements progress clockwise around the central pixel starting from the 3 o'clock position
	//TODO: see if intelligently selecting just 6 pixels to read based on gradient angle shows any significant benefit over 
	// the current method of reading all 8 adjacent into an array and indexing
	union l_c8 neighbors;
	neighbors.c.s0 = read_imagei(iC1_edge_image, samp, coords + (int2)(1,0)).x;
	neighbors.c.s1 = read_imagei(iC1_edge_image, samp, coords + 1).x;
	neighbors.c.s2 = read_imagei(iC1_edge_image, samp, coords + (int2)(0,1)).x;
	neighbors.c.s3 = read_imagei(iC1_edge_image, samp, coords + (int2)(-1,1)).x;
	neighbors.c.s4 = read_imagei(iC1_edge_image, samp, coords - (int2)(1,0)).x;
	neighbors.c.s5 = read_imagei(iC1_edge_image, samp, coords - 1).x;
	neighbors.c.s6 = read_imagei(iC1_edge_image, samp, coords - (int2)(0,1)).x;
	neighbors.c.s7 = read_imagei(iC1_edge_image, samp, coords - (int2)(-1,1)).x;

	//maybe don't do switch since that's really bad branching wise, might not hurt much now due to processing sparsity but could be bad later
	//switch(grad_ang + 16 & 0xE0)	// find which 22.5 degree offset octant the angle is part of

	// convert gradient angle to index of bin corresponding to it, +22.5 degrees (+16) to round to nearest bin
	uchar grad_idx = (uchar)(grad_ang + 16) >> 5;
	// rotate the neighbors so that indexing is now relative to the gradient vector
	// rotation amount is (array element count (8) minus grad_idx) * 8 bits per byte
	neighbors.l = rotate(neighbors.l, (long)(8 * (8 - grad_idx)));

	// at least one neighbor on this side must be populated for this to potentially count as a start
	if(!(neighbors.l & 0x00000000FFFFFF00))
		return;
	
	// need to figure which of the 3 cells was occupied, with priority to most close to the normal vector
	uchar rel_idx = 2;	// set to default option of index corresponding to +90 off of gradient as it's most likely to be occupied
	if(!neighbors.c.s2)	// if most normal cell isn't occupied,
	{
		// prioritize checking the next nearest angle to +90
		char dir = ((char)(grad_idx << 5) - grad_ang) < 0 ? -1 : 1;	// sign of the difference determines which bin to check next
		rel_idx += dir;
		if(!neighbors.a[rel_idx & 7])	// if it's still not occupied, we guessed wrong and it's the opposite direction
			rel_idx -= 2*dir;
	}
	
	uchar out_data = ((rel_idx + grad_idx -2) & 7) | 0xF0;	// convert from normal-relative index to 0 deg-relative index and OR occupancy flag/padding
	union i_c4 diffs, mask;
	// all neighbors on this side must be empty for this to count as a start
	switch(neighbors.l & 0xFFFFFF0000000000)	// switch used instead of if so that we can break out to end of block
	{
	default:
		// or in the case of a sharp corner, there must be significantly different angles
		diffs.c = neighbors.c.hi;
		mask.i = diffs.i & 0x01010100;
		mask.i = (mask.i << 8) - mask.i;
		diffs.uc = abs(diffs.c - grad_ang);
		diffs.i &= mask.i;
		if(any(diffs.uc >= (uchar)20))
		{
			out_data |= 8;	//set start flag
			break;
		}

		// or in the special corner case of a closed loop, the angle must go from negative to positive
		//NOTE: could also be done on gray code in same or fewer steps, ie shift-XOR, mask, ==
		if(grad_ang <= 64)	// positive pass check, done first to hopefully skip more branching
			break;
		
		/*
		// need to figure which of the 3 cells was occupied, with priority to most close to the normal vector
		rel_idx = 6;	// set to default option of index corresponding to -90 off of gradient as it's most likely to be occupied
		if(!neighbors.c.s6)	// if most normal cell isn't occupied,
		{
			// prioritize checking the next nearest angle to -90
			char dir = ((char)(grad_idx << 5) - grad_ang) < 0 ? -1 : 1;	// sign of the difference determines which bin to check next
			rel_idx += dir;
			if(!neighbors.a[rel_idx & 7])	// if it's still not occupied, we guessed wrong and it's the opposite direction
				rel_idx -= 2*dir;
		}*/

//		neighbors.l &= 0xFFFFFFFF000000FF;
		if(!any(neighbors.c < (char)-64))	// negative pass check
			break;
	case 0:	// no neighbors existed OR intentional fall through
		out_data |= 8;	//set start flag
	}

	write_imageui(uc1_starts_image, coords, out_data);
}