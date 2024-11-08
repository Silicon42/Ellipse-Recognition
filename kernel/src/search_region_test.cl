// Debugging kernel that mimics the processing of a single primary arc in the same way as arc_adj_matrix.cl
// and highlights regions excluded by checks, un-highlighted region is the final search area
#include "math_helpers.cl"

#define POKE_COORDS (int2)(96,91)

kernel void search_region_test(
	read_only image2d_t iC2_line_data,
	read_only image2d_t uc4_trace_image,
	write_only image2d_t uc4_debug)
{
	const int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uint4 color = read_imageui(uc4_trace_image, coords);	//base color inherited from the retrace, if it has any
	if(color.x | color.y)
	{
		write_imageui(uc4_debug, coords, color);
		return;
	}
	color = -1;

	int2 A_start = POKE_COORDS;

	int2 A_end_offset = read_imagei(iC2_line_data, A_start).lo;
	int2 A_end = A_start + A_end_offset;
	uint worst_dist2 = mag2_2d_i(A_end_offset);

//From the loop
	// check which location to evaluate for adjacency
	int2 B_start = coords;		// start of arc B as float2
	int2 A_to_B = B_start - A_end;	// vector from end of segment A to start of segment B
	uint dist2 = mag2_2d_i(A_to_B);
	// if it's at or above the max search radius away from the end,
	// skip it, it's not likely part of the same ellipse,
	// also prevents it from including itself
	if(dist2 < worst_dist2)
	{
		color.x = 0;
	}

	// if start of segment B isn't forward of the end of segment A,
	// A_to_B will have a component against the direction of A_end_offset
	// so dot product will be negative, indicating it should be skipped
	if(dot_2d_i(A_end_offset, A_to_B) >= 0)
	{
		color.z = 0;
	}
	
	if(cross_2d_i(A_end_offset, A_to_B) >= 0)
	{
		color.y = 0;
	}

	write_imageui(uc4_debug, coords, color);
}