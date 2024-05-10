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
	neighbors.c.s0 = read_imagei(iC1_src_image, coords + (int2)(1,0)).x;
	neighbors.c.s1 = read_imagei(iC1_src_image, coords + 1).x;
	neighbors.c.s2 = read_imagei(iC1_src_image, coords + (int2)(0,1)).x;
	neighbors.c.s3 = read_imagei(iC1_src_image, coords + (int2)(-1,1)).x;
	neighbors.c.s4 = read_imagei(iC1_src_image, coords - (int2)(1,0)).x;
	neighbors.c.s5 = read_imagei(iC1_src_image, coords - 1).x;
	neighbors.c.s6 = read_imagei(iC1_src_image, coords - (int2)(0,1)).x;
	neighbors.c.s7 = read_imagei(iC1_src_image, coords - (int2)(-1,1)).x;

	//maybe don't do switch since that's really bad branching wise, might not hurt much now due to processing sparsity but could be bad later
	//switch(grad_ang + 16 & 0xE0)	// find which 22.5 degree offset octant the angle is part of

	// convert gradient angle to index of bin corresponding to it, +22.5 degrees (+16) to round to nearest bin
	uchar grad_idx = (grad_ang + 16) >> 5;
	neighbors.l = rotate(neighbors.l, (long)grad_idx);//assuming the rotation is a right rotation

	// all neighbors on this side must be empty for this to count as a start
	if(neighbors.l & 0x0101010000000000)
	{
		// or in the special corner case of a closed loop, the angle must go from low negative to low positive
		//NOTE: could also be done on gray code in same or fewer steps, ie shift-XOR, mask, ==
		if(!(grad_ang > 0 && grad_ang < 64))	//low positive check
			return;
		
		// need 3 checks since we don't know which of the cells was occupied,
		// can't be switch-case since multiple (should be no more than 2) can be occupied
		char i = 7;
		if(neighbors.c.s6)
			i = 6;
		else if(neighbors.c.s5)
			i = 5;
		
		if(!(neighbors.a[i] < 0 && neighbors.a[i] > -64))
			return;
		
		//TODO: maybe add this connection as part of the segment, would require additional processing
	}

	//norm_idx = (norm_idx + 4) & 7;	// flip to opposite normal index
	// at least one neighbor on this side must be populated for this to count as a start
	if(!(neighbors.l & 0x0000000001010100))
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