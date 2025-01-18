#ifndef NEIGHBOR_UTILS_CL
#define NEIGHBOR_UTILS_CL

#include "samplers.cl"
#include "cast_helpers.cl"

inline char8 read_neighbors_ccw(read_only image2d_t ic1_edge_image, const int2 coords)
{
	char8 neighbors;
	neighbors.s0 = read_imagei(ic1_edge_image, clamped, coords + (int2)( 1, 0)).x;
	neighbors.s1 = read_imagei(ic1_edge_image, clamped, coords + (int2)( 1,-1)).x;
	neighbors.s2 = read_imagei(ic1_edge_image, clamped, coords + (int2)( 0,-1)).x;
	neighbors.s3 = read_imagei(ic1_edge_image, clamped, coords - 1).x;
	neighbors.s4 = read_imagei(ic1_edge_image, clamped, coords + (int2)(-1, 0)).x;
	neighbors.s5 = read_imagei(ic1_edge_image, clamped, coords + (int2)(-1, 1)).x;
	neighbors.s6 = read_imagei(ic1_edge_image, clamped, coords + (int2)( 0, 1)).x;
	neighbors.s7 = read_imagei(ic1_edge_image, clamped, coords + 1).x;
	return neighbors;
}

inline char8 read_neighbors_cw(read_only image2d_t ic1_edge_image, const int2 coords)
{
	char8 neighbors;
	neighbors.s0 = read_imagei(ic1_edge_image, clamped, coords + (int2)( 1, 0)).x;
	neighbors.s1 = read_imagei(ic1_edge_image, clamped, coords + 1).x;
	neighbors.s2 = read_imagei(ic1_edge_image, clamped, coords + (int2)( 0, 1)).x;
	neighbors.s3 = read_imagei(ic1_edge_image, clamped, coords + (int2)(-1, 1)).x;
	neighbors.s4 = read_imagei(ic1_edge_image, clamped, coords + (int2)(-1, 0)).x;
	neighbors.s5 = read_imagei(ic1_edge_image, clamped, coords - 1).x;
	neighbors.s6 = read_imagei(ic1_edge_image, clamped, coords + (int2)( 0,-1)).x;
	neighbors.s7 = read_imagei(ic1_edge_image, clamped, coords + (int2)( 1,-1)).x;
	return neighbors;
}

inline long get_occupancy_mask(long neighbors)
{
	long occupancy = neighbors & 0x0101010101010101;	//extract the occupancy flags
	return (occupancy << 8) - occupancy;	// convert flags to mask
}

// returns a mask of only angle deltas within +/-90 degrees
inline long is_diff_small(uchar8 diff, long occupancy)
{
	union l_conv is_diff_small;
	is_diff_small.c = diff < (uchar)64;
	return is_diff_small.l & occupancy;
}

// get the indices of the 2 smallest values, arbitrary order
inline union s_conv select_min_2(const uchar* comp)
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

#endif//NEIGHBOR_UTILS_CL