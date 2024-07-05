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

union i_c4{
	int i;
	char4 c;
	uchar4 uc;
	uchar uca[4];
};

kernel void find_segment_starts(read_only image2d_t iC1_edge_image, write_only image2d_t uc1_starts_image)
{
	//const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
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
	// order is reversed from typical since rotate() is a left rotate only
	union l_c8 neighbors;
	neighbors.c.s0 = read_imagei(iC1_edge_image, clamped, coords + (int2)(1, 0)).x;
	neighbors.c.s1 = read_imagei(iC1_edge_image, clamped, coords + (int2)(1,-1)).x;
	neighbors.c.s2 = read_imagei(iC1_edge_image, clamped, coords - (int2)(0, 1)).x;
	neighbors.c.s3 = read_imagei(iC1_edge_image, clamped, coords - 1).x;
	neighbors.c.s4 = read_imagei(iC1_edge_image, clamped, coords - (int2)(1, 0)).x;
	neighbors.c.s5 = read_imagei(iC1_edge_image, clamped, coords - (int2)(1,-1)).x;
	neighbors.c.s6 = read_imagei(iC1_edge_image, clamped, coords + (int2)(0, 1)).x;
	neighbors.c.s7 = read_imagei(iC1_edge_image, clamped, coords + 1).x;

	// convert gradient angle to index of bin corresponding to it, +22.5 degrees (+16) to round to nearest bin
	uchar grad_idx = (uchar)(grad_ang + 16) >> 5;
	// rotate the neighbors so that indexing is now relative to the gradient vector
	// rotation amount is (array element count (8) minus grad_idx) * 8 bits per byte
	neighbors.l = rotate(neighbors.l, (long)(8 * grad_idx));

	if(all(coords == (int2)(587, 708)))
		printf("0x%.8X %X %.8X\n", neighbors.i.hi, (uchar)grad_ang, neighbors.i.lo);


	long occupancy = neighbors.l & 0x0101010101010101;
	occupancy = (occupancy << 8) - occupancy;
	union l_c8 diff, is_diff_small;
	diff.uc = abs(neighbors.c - grad_ang);
	is_diff_small.c = diff.uc < (uchar)32;
	is_diff_small.l &= occupancy;
	uchar rel_idx;
	// exactly one neighbor on this side must be populated for this to potentially count as a start
	switch(is_diff_small.i.lo & 0xFFFFFF00)
	{
	default:
		return;
	case 0x000000FF:
		rel_idx = 0;
		break;
	case 0x0000FF00:
		rel_idx = 1;
		break;
	case 0x00FF0000:
		rel_idx = 2;
		break;
	case 0xFF000000:
		rel_idx = 3;
	}
	
	// all neighbors on this side must not be similar angles for this to count as a start
	if(is_diff_small.i.hi)
	{
		// or in the special corner case of a closed loop, we add a start at the zero crossing
		if(grad_ang >= 32 || grad_ang < -32)
			return;
		union i_c4 roll_over;
		roll_over.c = grad_ang ^ neighbors.c.hi;
		roll_over.i &= 0xE0E0E000 & is_diff_small.i.hi;
		if(all(roll_over.c != (char)0xE0))
			return;
	}

	uchar out_data = ((rel_idx + grad_idx -2) & 7) | 0xF8;	// convert from normal-relative index to 0 deg-relative index and OR occupancy flag/padding
	write_imageui(uc1_starts_image, coords, out_data);
}