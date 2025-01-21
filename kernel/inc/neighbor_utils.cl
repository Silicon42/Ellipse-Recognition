#ifndef NEIGHBOR_UTILS_CL
#define NEIGHBOR_UTILS_CL

#include "samplers.cl"
#include "cast_helpers.cl"

#define OCCUPANCY_FLAGS	0x0101010101010101

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

#endif//NEIGHBOR_UTILS_CL