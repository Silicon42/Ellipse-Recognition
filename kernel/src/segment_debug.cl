// Debug kernel that combines the outputs of canny_short, reject_intersections, find_segment_starts, and construct_segments
// for visualization purposes

__kernel void segment_debug(read_only image2d_t iC1_canny, read_only image2d_t iC1_reject_isect, image2d_t iC1_seg_start, write_only image2d_t iC4_dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	int4 out = -1;
	out.z = read_imagei(iC1_reject_isect, coords).x ? -1 : 0;	// sets blue channel if edge passed intersection rejection
	out.x = (!out.z && read_imagei(iC1_canny, coords).x) ? -1 : 0;	// sets red channel if edge didn't pass intersection rejection
	out.y = read_imagei(iC1_seg_start, coords).x ? -1 : 0;	// sets green channel (in combo with blue) if pixel classified as a segment start

	write_imagei(iC4_dst_image, coords, out);
}