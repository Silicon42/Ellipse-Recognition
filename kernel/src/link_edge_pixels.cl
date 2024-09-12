#include "vec_sum_reduce.cl"
#include "neighbor_utils.cl"

__kernel void link_edge_pixels(read_only image2d_t iC1_grad_ang, write_only image2d_t uc1_cont)
{
	//const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	//needed so that edge
	const int2 coords = (int2)(get_global_id(0), get_global_id(1));

	//TODO: see if reading as 32 bit values and casting has better performance since it would cut out the conversion step
	char grad_ang = read_imagei(iC1_grad_ang, coords).x;
	// magnitude channel used as bool for occupancy
	// if gradient angle == 0, it wasn't set in canny_short because even 0 should have the occupancy flag set,
	// therefore this work item isn't on an edge and can exit early, vast majority exits here
	if(!grad_ang)
		return;

	union l_conv neighbors;
	neighbors.c = read_neighbors_cw(iC1_grad_ang, coords);

	// reject orphan edge pixels here, technically intersection rejection also creates some more but I don't have a good way
	// to do that in find_segment_starts.cl without adding an additional output argument and it's not super important, just
	// a small optimization/cleanup step for nicer debug output
	if(!neighbors.l)
		return;

	long occupancy = get_occupancy_mask(neighbors.l);
	union l_conv diff;
	diff.uc = abs(neighbors.c - grad_ang);
	long is_diff_small_mask = is_diff_small(diff.uc, occupancy);
	// reject pixels that are exclusively surrounded by pixels that have high angular differences relative to them (>= +/-45 degrees)
	// these are typically noise or sharp corners that would be better picked up individually as separate edges
	if(!is_diff_small_mask)
		return;

	// any instance where there are more than 2 similarly angled pixels adjacent to a pixel after thinning could result in a
	// branch starting and since it can't be determined at this level which will be taken
	uchar cont_data = 0;
	//uchar grad_idx = grad_ang >> 5;
	if(popcount(is_diff_small_mask) == 8)	// only one continuation
	{
		char index = 7 - (clz(is_diff_small_mask) >> 3);
		// determine if this is a right or left continuation based on if the difference
		cont_data = (grad_ang - (index << 5) < 0) ? index | 8 : (index << 5) | 16;
	}
	else
	{
		// get the indices of the 2 neighbors closest in angle to the current pixel
		union l_conv to_compare;
		to_compare.l = diff.l | ~is_diff_small_mask;
		union s_conv index2 = select_min_2(to_compare.uca);
		// we then need the average angle between those indices +90 degrees to use as reference
		// in order for this to be treated correctly across the over/underflow boundary this must
		// be acheived by a half difference added to the index
		char ref_ang = (index2.c.x + index2.c.y - 4) << 4;
		char order = (index2.c.x > index2.c.y) ^ ((char)(grad_ang - ref_ang) < 0);	// done as subtraction to allow roll over
		// apparently subtraction and presumably other math operations automatically promotes the involved operands to a type
		// larger than a char despite all of them being chars which causes it to fail to roll-over without a cast so that's why
		// the above line needs a cast before the comparison happens
		cont_data = 0x18 | index2.ca[!order] | (index2.ca[order] << 5);
	}

	write_imageui(uc1_cont, coords, (int)cont_data);
}
// left-over code that I might need later elsewhere
/*
		// angle smoothing, helps reduce angular noise
		diff.l &= is_diff_small_mask.l;
		// divisor determines weighting,
		// /2: avg of adjacent angles,
		// /3: avg of adjacent and this pixel,
		// /4+: avg of adjacent and this pixel with weighting towards this pixel
		grad_ang = (sum_reduce(diff.c)/4 + grad_ang) | 1;

*/