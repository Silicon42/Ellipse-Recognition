// Kernel meant to remove interstections from the edge map as they are problematic for processing segments since they would
// require it to branch
#include "samplers.cl"
#include "cast_helpers.cl"

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

	union l_conv neighbors;
	neighbors.c.s0 = read_imagei(iC1_src_image, clamped, coords + (int2)( 1, 0)).x;
	neighbors.c.s1 = read_imagei(iC1_src_image, clamped, coords + 1).x;
	neighbors.c.s2 = read_imagei(iC1_src_image, clamped, coords + (int2)( 0, 1)).x;
	neighbors.c.s3 = read_imagei(iC1_src_image, clamped, coords + (int2)(-1, 1)).x;
	neighbors.c.s4 = read_imagei(iC1_src_image, clamped, coords + (int2)(-1, 0)).x;
	neighbors.c.s5 = read_imagei(iC1_src_image, clamped, coords - 1).x;
	neighbors.c.s6 = read_imagei(iC1_src_image, clamped, coords + (int2)( 0,-1)).x;
	neighbors.c.s7 = read_imagei(iC1_src_image, clamped, coords + (int2)( 1,-1)).x;

	// reject orphan edge pixels here, technically intersection rejection also creates some more but I don't have a good way
	// to do that in find_segment_starts.cl without adding an additional output argument and it's not super important, just
	// a small optimization/cleanup step for nicer debug output
	if(!neighbors.l)
		return;

	long occupancy = neighbors.l & 0x0101010101010101;	//extract just the occupancy flags
	occupancy = (occupancy << 8) - occupancy;	// convert flags to mask
	union l_conv diff, is_diff_small;
	diff.uc = abs(neighbors.c - grad_ang);
	is_diff_small.c = diff.uc < (uchar)32;
	is_diff_small.l &= occupancy;
	// reject pixels that are exclusively surrounded by pixels that have high angular differences relative to them (>= +/-45 degrees)
	// these are typically noise or sharp corners that would be better picked up individually as separate edges
	//TODO: verify if this condition is still neccessary
	if(!is_diff_small.l)
		return;

	if(all(coords == (int2)(447,83)))
		printf("%i %i %i %i %i %i %i %i\n", neighbors.c.s0, neighbors.c.s1, neighbors.c.s2, neighbors.c.s3, neighbors.c.s4, neighbors.c.s5, neighbors.c.s6, neighbors.c.s7);
	// any instance where there are more than 2 similarly angled pixels adjacent to a pixel after thinning could result in a
	// branch starting and since it can't be determined at this level which will be taken, it's best to invalidate the pixel as
	// an intersection even if it could be retained
	uchar near_count = popcount(is_diff_small.l);
	if(near_count > 16)
		return;

	write_imagei(iC1_dst_image, coords, (int)grad_ang);
}