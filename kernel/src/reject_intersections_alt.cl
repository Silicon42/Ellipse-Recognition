// Kernel meant to remove interstections from the edge map as they are problematic for processing segments since they would
// require it to branch


union l_c8{
	long l;
	char8 c;
	uchar8 uc;
	char a[8];
};

__kernel void reject_intersections_alt(read_only image2d_t iC1_src_image, write_only image2d_t iC1_dst_image)
{
	//const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	//needed so that edge
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	//TODO: see if reading as 32 bit values and casting has better performance since it would cut out the conversion step
	char grad_ang = read_imagei(iC1_src_image, coords).x;
	// magnitude channel used as bool for occupancy
	// if gradient angle == 0, it wasn't set in canny_short because even 0 should have the occupancy flag set,
	// therefore this work item isn't on an edge and can exit early, vast majority exits here
	if(!grad_ang)
		return;

	union l_c8 neighbors;
	neighbors.c.s0 = read_imagei(iC1_src_image, clamped, coords + (int2)(1,0)).x;
	neighbors.c.s1 = read_imagei(iC1_src_image, clamped, coords + 1).x;
	neighbors.c.s2 = read_imagei(iC1_src_image, clamped, coords + (int2)(0,1)).x;
	neighbors.c.s3 = read_imagei(iC1_src_image, clamped, coords + (int2)(-1,1)).x;
	neighbors.c.s4 = read_imagei(iC1_src_image, clamped, coords - (int2)(1,0)).x;
	neighbors.c.s5 = read_imagei(iC1_src_image, clamped, coords - 1).x;
	neighbors.c.s6 = read_imagei(iC1_src_image, clamped, coords - (int2)(0,1)).x;
	neighbors.c.s7 = read_imagei(iC1_src_image, clamped, coords - (int2)(-1,1)).x;

	// reject orphan edge pixels here, technically intersection rejection also creates some more but I don't have a good way
	// to do that in find_segment_starts.cl without adding an additional output argument and it's not super important, just
	// a small optimization/cleanup step for nicer debug output
	if(!neighbors.l)
		return;

	long occupancy = neighbors.l & 0x0101010101010101;	//extract just the occupancy flags
	long occupancy_mask = (occupancy << 8) - occupancy;	// convert flags to mask
	union l_c8 diff, is_diff_big;
	diff.uc = abs(neighbors.c - grad_ang);
	is_diff_big.c = diff.uc >= (uchar)32;
	is_diff_big.l &= occupancy_mask;
	// reject pixels that are exclusively surrounded by pixels that have high angular differences relative to them (>= +/-45 degrees)
	// these are typically noise or sharp corners that would be better picked up individually as separate edges
	if(is_diff_big.l == occupancy_mask)
		return;

	uchar near_count = popcount(~is_diff_big.l & occupancy_mask);
	if(near_count > 24)
		return;
	if(is_diff_big.l && near_count != 16)
		return;
	//NOTE: This block is wrong because in some extreme situations, an edge can appear as a constant gradient for multiple pixels
	// without a single clearly defined max, this is most common in simple computer generated images but could theoretically happen
	// in real life as well
	//any situation with more than 4 neighbors must be an intersection, so reject
	//char neighbor_cnt = popcount(occupancy);
	//if(neighbor_cnt >= 3)
	//	return;
/*
	// AND-ing a rotation by 1 position of the occupancy with itself results in a boolean vector representing
	// neighbors that are also neighbors of occupied cells.
	long mutual_neighbors = occupancy & rotate(occupancy, 8L);
	// convert the boolean vector to a mask of just the angle portion corresponding to valid differences of mutually adjacent cells
	mutual_neighbors = (mutual_neighbors << 8) - mutual_neighbors;

	// the subtract as a long works because all borrows come from the occupancy flags leaving valid angles untouched and invalid
	// ones get masked out anyway, this means that it works in parallel regardless of if there is hardware vector support or not,
	// this may or may not be faster than proper vector operations on systems with hardware vector support on a case by case basis
	union l_c8 mutual_diff;
	mutual_diff.l = mutual_neighbors & (neighbors.l - rotate(neighbors.l, 8L));

	union l_c8 is_invalid;
	//TODO: this threshold might need to be widened or shrunk depending on gradient finding method, currently assumes most angle divergence possible while still belonging to the same arc is 45 degrees
	is_invalid.c = abs(mutual_diff.c) > (uchar)64;	// if absolute divergence in angle of adjacent neighbors is more than 90 degrees (90/360 : 1/4 : 64/256)
	if(is_invalid.l)
	{
	//	printf("invalid: 0x %08X %08X, diff: 0x %08X %08X, mask: 0x %08X %08X, values: 0x %08X %08X\n", is_invalid.l >> 32, is_invalid.l, mutual_diff.l >> 32, mutual_diff.l, mutual_neighbors >> 32, mutual_neighbors, neighbors.l >> 32, neighbors.l);
		return;
	}*/
	write_imagei(iC1_dst_image, coords, (int)grad_ang);
}