// Debug kernel that combines the outputs of canny_short, reject_intersections, find_segment_starts, and construct_segments
// for visualization purposes

__kernel void starts_debug(read_only image2d_t iC1_canny, read_only image2d_t iC1_reject_isect, read_only image2d_t uc1_seg_start, write_only image2d_t uc4_dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	uint4 out = 0;
	out.w = -1;
	out.y = read_imageui(uc1_seg_start, coords).x ? -1 : 0;			// sets green channel if pixel classified as a segment start
	out.z = read_imagei(iC1_reject_isect, coords).x ? -1 : 0;		// sets blue channel if edge passed intersection rejection
	out.x = (!out.z && read_imagei(iC1_canny, coords).x) ? -1 : 0;	// sets red channel if edge didn't pass intersection rejection

	write_imageui(uc4_dst_image, coords, out);
}