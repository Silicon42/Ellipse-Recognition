// Kernel meant to select intial line/arc segment starting points in a non-max
// suppressed edge image (such as after Canny) and hash them into a 1D array.
// Since starts should be extremely sparse, a sufficiently large hash table should
// have few collisions but still take up less space and have better access patterns
// than operating on the whole image. If it can't be made sufficiently big enough,
// then the secondary output can be used to confirm remaining starts after transformation
// back into uncompressed space and processing on those remaining can be doen in a
// 2nd(or more) pass
/*
union l_c8{
	long l;
	char8 c;
	char a[8];
};
*/
__kernel void find_segment_starts(read_only image2d_t iC1_src_image, write_only image2d_t iC1_dst_image)
{
	const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	char grad_ang = read_imagei(iC1_src_image, coords).x;

	// magnitude channel used as bool for occupancy
	// if gradient angle == 0, it wasn't set in canny_short because even 0 should have the occupancy flag set,
	// therefore this work item isn't on an edge and can exit early, vast majority exits here
	if(!grad_ang)
		return;

	// read adjacent pixels into an array such that elements progress clockwise around the central pixel starting from the 3 o'clock position
	//TODO: see if intelligently selecting just 6 pixels to read based on gradient angle shows any significant benefit over 
	// the current method of reading all 8 adjacent into an array and indexing
	union l_c8 neighbors;
	neighbors.c.s0 = read_imagei(iC1_src_image, samp, coords + (int2)(1,0)).x;
	neighbors.c.s1 = read_imagei(iC1_src_image, samp, coords + 1).x;
	neighbors.c.s2 = read_imagei(iC1_src_image, samp, coords + (int2)(0,1)).x;
	neighbors.c.s3 = read_imagei(iC1_src_image, samp, coords + (int2)(-1,1)).x;
	neighbors.c.s4 = read_imagei(iC1_src_image, samp, coords - (int2)(1,0)).x;
	neighbors.c.s5 = read_imagei(iC1_src_image, samp, coords - 1).x;
	neighbors.c.s6 = read_imagei(iC1_src_image, samp, coords - (int2)(0,1)).x;
	neighbors.c.s7 = read_imagei(iC1_src_image, samp, coords - (int2)(-1,1)).x;

	//maybe don't do switch since that's really bad branching wise, might not hurt much now due to processing sparsity but could be bad later
	//switch(grad_ang + 16 & 0xE0)	// find which 22.5 degree offset octant the angle is part of

	// convert gradient angle to index of bin corresponding to it, +22.5 degrees (+16) to round to nearest bin
	uchar grad_idx = (uchar)(grad_ang + 16) >> 5;
	// rotate the neighbors so that indexing is now relative to the gradient vector
	// rotation amount is index count (8) minus grad_idx * 8 bits per byte
	neighbors.l = rotate(neighbors.l, (long)(8 * (8 - grad_idx)));

	// all neighbors on this side must be empty for this to count as a start
	if(neighbors.l & 0xFFFFFF0000000000)
	{
		// or in the special corner case of a closed loop, the angle must go from negative to positive
		//NOTE: could also be done on gray code in same or fewer steps, ie shift-XOR, mask, ==
		//FIXME: sometimes this leads to short near vertical segments consisting of mostly segment starts b/c they'll be generated
		// right next to normal starts. This either needs special handling later or here
		//FIXME: There seems to be some rare corner case where this fails to pick up a start on a loop, currently input6.png shows this best
		// Revisit once I have more info and insight
		if(grad_ang < 64)	// positive pass check, done first to hopefully skip more branching
			return;
		
		// need to figure which of the 3 cells was occupied,
		// can't be switch-case since multiple (should ideally be no more than 2) can be occupied
		char i = 7;
		if(neighbors.c.s6)	// highest priority to the nearest to 90, most likely to be occupied
			i = 6;
		else if(neighbors.c.s5)	// 2nd highest to ccw of 90 by convention
			i = 5;
		//else i = 7;	defaults to cw of 90 because if it wasn't the first 2 it must be this one

		if(neighbors.a[i] > -64)	// negative pass check
			return;
	}

	//norm_idx = (norm_idx + 4) & 7;	// flip to opposite normal index
	// at least one neighbor on this side must be populated for this to count as a start
	if(!(neighbors.l & 0x00000000FFFFFF00))
		return;
	
	// need 3 checks same as above since we don't know which of the cells was occupied
	char i = 0b10;
	if(neighbors.c.s2)
		i = 0b11;
	else if(neighbors.c.s1)
		i = 0b01;
	
	//neighbors.a[i] - grad_ang;
	//TODO: coords should get updated to the coord of the segment continuation here for processing purposes

	write_imagei(iC1_dst_image, coords, i<<6);
}