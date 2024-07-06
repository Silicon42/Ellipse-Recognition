// Debug kernel that combines the outputs of canny_short, reject_intersections, find_segment_starts, and construct_segments
// for visualization purposes

__kernel void starts_debug(read_only image2d_t iC1_canny, read_only image2d_t iC1_reject_isect, read_only image2d_t uc1_seg_start, write_only image2d_t uc4_dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	uint4 out = 0;
	out.w = -1;
	char grad_ang = read_imagei(iC1_reject_isect, coords).x;
	if(grad_ang & 1)
	{
		out.z = grad_ang | 0x1F;	// sets blue channel if edge passed intersection rejection
		if(read_imageui(uc1_seg_start, coords).x & 8)
			out.y = grad_ang | 0x1F;	// sets green channel if pixel classified as a segment start
	}
	else
		out.y = (read_imageui(uc1_seg_start, coords).x & 8) ? -1 : 0;	// sets full green channel if pixel classified as a segment start but not edge
	out.x = (!out.z && read_imagei(iC1_canny, coords).x) ? 127 : 0;	// sets red channel if edge didn't pass intersection rejection

	write_imageui(uc4_dst_image, coords, out);
}