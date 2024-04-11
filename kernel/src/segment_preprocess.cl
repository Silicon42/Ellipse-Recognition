// Kernel meant to select intial line/arc segment starting points in a non-max
// suppressed edge image (such as after Canny) and hash them into a 1D array.
// Since starts should be extremely sparse, a sufficiently large hash table should
// have few collisions but still take up less space and have better access patterns
// than operating on the whole image. If it can't be made sufficiently big enough,
// then the secondary output can be used to confirm remaining starts after transformation
// back into uncompressed space and processing on those remaining can be doen in a
// 2nd(or more) pass

//#include "colorize.h"

__kernel void segment_preprocess(read_only image2d_t us2_src_image, write_only image2d_t fc4_dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	uint2 grad = read_imageui(us2_src_image, coords).hi;
	// if gradient == 0, this work item isn't on an edge and can exit early
	if(grad.y == 0)
		return;

	ushort8 neighbors;
	neighbors.s0 = read_imageui(us2_src_image, coords + (int2)(1,0)).x;
	neighbors.s1 = read_imageui(us2_src_image, coords + 1).x;
	neighbors.s2 = read_imageui(us2_src_image, coords + (int2)(0,1)).x;
	neighbors.s3 = read_imageui(us2_src_image, coords + (int2)(-1,1)).x;
	neighbors.s4 = read_imageui(us2_src_image, coords + (int2)(-1,0)).x;
	neighbors.s5 = read_imageui(us2_src_image, coords - 1).x;
	neighbors.s6 = read_imageui(us2_src_image, coords + (int2)(0,-1)).x;
	neighbors.s7 = read_imageui(us2_src_image, coords + (int2)(1,-1)).x;

}