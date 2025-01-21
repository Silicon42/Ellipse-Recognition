#include "neighbor_utils.cl"
#include "link_macros.cl"

// convert flags to mask
inline long get_occupancy_mask(long flags)
{	return (flags << 8) - flags;	}

// returns the index in the neighbors array of the corner that is the minimum
inline uchar select_min_corner(uchar4 comp)
{
	union s_conv sel2;
	sel2.c = comp.hi < comp.lo;
	uchar2 comp2 = select(comp.lo, comp.hi, sel2.uc);
	char sel1 = comp2.y < comp2.x;
	return (sel2.ca[sel1] & 4) + sel1 * 2 + 1;
}

inline union s_conv select_min_2(uchar const* comp)
{
	union s_conv sel_min;
	sel_min.c = (char2)(0, 1);
	uchar min_pos = comp[0] > comp[1];
	for(uchar i = 2; i < 8; ++i)
	{
		if(comp[i] < comp[sel_min.ca[!min_pos]])
		{
			sel_min.ca[!min_pos] = i;	// write the min/2nd min over where the old 2nd min was
			// if the current position meets or beats the current minimum
			if(comp[i] <= comp[sel_min.ca[min_pos]])
				min_pos = !min_pos;	// toggle minimum slot, this leaves the old minimum as the 2nd min
		}
	}

	return sel_min;
}

__kernel void link_edge_pixels(
	read_only image2d_t ic1_grad_ang,
	write_only image2d_t uc1_cont)
{
	const int2 coords = (int2)(get_global_id(0), get_global_id(1));

	char grad_ang = read_imagei(ic1_grad_ang, coords).x;
	// if gradient angle == 0, it wasn't set in canny_short because even 0 should have the occupancy flag set,
	// therefore this work item isn't on an edge and can exit early, vast majority exits here
	if(!grad_ang)
		return;

	union l_conv neighbors;
	neighbors.c = read_neighbors_cw(ic1_grad_ang, coords);

	// reject orphan edge pixels here, technically intersection rejection also creates some more but I don't have a good way
	// to do that in find_segment_starts.cl without adding an additional output argument and it's not super important
	//NOTE: technically the small difference mask check would also catch these so it might be better for performance to just remove this check
	if(!neighbors.l)
		return;

	long occ_flags = neighbors.l & OCCUPANCY_FLAGS;
	long occ_mask = get_occupancy_mask(occ_flags);
	union l_conv diff, is_diff_small_mask;
	diff.uc = abs(neighbors.c - grad_ang);
	// all values that check diff expect are looking for min so set unoccupied slots to 255
	diff.l |= ~occ_mask;
	//TODO: subsequent popcounts might benefit from a flags only version depending on their implementation
	is_diff_small_mask.c = diff.uc < (uchar)64;
	// reject pixels that are exclusively surrounded by pixels that have high angular differences relative to them (>= +/-90 degrees)
	// these are typically noise or sharp corners that would be better picked up individually as separate edges
	if(!is_diff_small_mask.l)
		return;

	uchar cont_data = 0;

	union s_conv indices;
	uchar * index = indices.uca;
	long adj_small_mask;
	char adj_small_pcnt, small_pcnt = popcount(is_diff_small_mask.l);
	if(all(coords == (int2)(430, 161)))
		printf("%v8hhx	%i\n", is_diff_small_mask.uc, small_pcnt);
	switch(small_pcnt)
	{
	case 8:		// only 1 continuation
		index[0] = ctz(is_diff_small_mask.l) >> 3;
		// determine if this is a right or left continuation based on if the difference between the continuation direction index
		// and the angle of the current pixel is positive or negative
		cont_data = (grad_ang - (index[0] << 5) < 0) ? index[0] | HAS_R_CONT : (index[0] << L_CONT_IDX_SHIFT) | HAS_L_CONT;
		write_imageui(uc1_cont, coords, (int)cont_data);
		return;
	default:	// more than 2 continuations...
		adj_small_mask = 0xFF00FF00FF00FF & is_diff_small_mask.l;
		// priority for continuations is given to face adjacent pixels
		adj_small_pcnt = popcount(adj_small_mask);
		if(adj_small_pcnt == 8)	// but only 1 face adjacent
		{
			uchar adj_idx = ctz(adj_small_mask) >> 3;	// priority to face adjacent
			//TODO: priority to non-adjacent to face adjacent
			// select the 2 corners not adjacent to the face adjacent one
			indices.uc = ((uchar2)(3, 5) + adj_idx) & (uchar)7;
			// replace whichever has a larger difference with the face adjacent one
			index[index[1] > index[0]] = adj_idx;
			break;
		}
		else	// either 2+ face adjacent or 3+ corner adjacent
		{
			if(adj_small_pcnt >= 16)	// 2+ face adjacent
				is_diff_small_mask.l = adj_small_mask;	// can safely ignore non-face-adjacent pixels
	if(all(coords == (int2)(430, 161)))
		printf("%v8hhx	%i\n", is_diff_small_mask.uc, adj_small_pcnt);
			if(adj_small_pcnt != 16)	// 3+ or 0(3+ corner adjacent) face adjacent, select min 2 from remaining
			{	// get the indices of the 2 neighbors closest in angle to the current pixel
				diff.l |= ~is_diff_small_mask.l;
				indices = select_min_2(diff.uca);
			}
			//else exactly 2 face adjacent, fall through to set index
		}
	case 16:	// only 2 continuations
		index[0] = ctz(is_diff_small_mask.l) >> 3;
		index[1] = 7 - (clz(is_diff_small_mask.l) >> 3);
	}
	// we then need the average angle between those indices +90 degrees to use as reference
	// in order for this to be treated correctly across the over/underflow boundary this must
	// be acheived by a half difference added to the index
	char ref_ang = (index[0] + index[1] - 4) << 4;
	char order = (index[0] > index[1]) ^ ((char)(grad_ang - ref_ang) < 0);	// done as subtraction to allow roll over
	// apparently subtraction and presumably other math operations automatically promotes the involved operands to a type
	// larger than a char despite all of them being chars which causes it to fail to roll-over without a cast so that's why
	// the above line needs a cast before the comparison happens
	cont_data = HAS_BOTH_CONT | index[!order] | (index[order] << L_CONT_IDX_SHIFT);
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