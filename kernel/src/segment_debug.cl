// Debug kernel that combines the outputs of canny_short, reject_intersections, find_segment_starts, and construct_segments
// for visualization purposes

__kernel void segment_debug(read_only image2d_t iC1_canny, read_only image2d_t iC1_reject_isect, read_only image2d_t uc1_seg_start, read_only image2d_t ui4_segments, read_only image2d_t uc1_seg_trace, write_only image2d_t uc4_dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	uint4 out = 0;
	out.w = -1;
	out.y =( read_imageui(uc1_seg_start, coords).x & 8) ? -1 : 0;	// sets green channel if pixel classified as a segment start
	if(!out.y)
		out.y = read_imagei(ui4_segments, coords).x ? 102 : 0;	// sets fraction of green channel if pixel was not a segment start but triggered a restart in arc_segments
	out.z = read_imageui(uc1_seg_trace, coords).x ? -1 : 0;	// sets blue channel if arc_segments algorithm visited the pixel
	if(!out.z)
		out.z = (!out.y && read_imagei(iC1_reject_isect, coords).x) ? 64 : 0;	// sets fraction of blue channel if edge passed intersection rejection and wasn't a start
	out.x = (!(out.z||out.y) && read_imagei(iC1_canny, coords).x) ? -1 : 0;	// sets red channel if edge didn't pass intersection rejection

	write_imageui(uc4_dst_image, coords, out);
}