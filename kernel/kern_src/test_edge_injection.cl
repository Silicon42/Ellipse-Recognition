// injects a series of 5 points in place of the line_segments kernel for testing the arc_builder code for correctness

kernel void test_edge_injection(
	write_only image1d_t is2_start_coords,
	write_only image2d_t ic2_line_data,
	write_only image1d_t us1_line_counts)
{
	//this set of test points should result in an ellipse with foci (123,83) and (37,-3)
	int2 test_points[5] = {20, (int2)(40,60), (int2)(60,80), (int2)(100,100), (int2)(120,100)};
	//this set of test points should result in an ellipse with foci (280.599915,-88.354492) and (59.400108,48.354485)
//	int2 test_points[5] = {(int2)(20,40), (int2)(40,80), (int2)(80,100), (int2)(140,100), (int2)(200,80)};
	//this set of test points should result in an hyperbola with foci (40*sqrt(2), 40*sqrt(2)) and (-40*sqrt(2), -40*sqrt(2))
//	int2 test_points[5] = {(int2)(160,10), (int2)(80,20), (int2)(40,40), (int2)(20,80), (int2)(10,160)};

	write_imagei(is2_start_coords, 0, (int4)(test_points[0], 0, -1));
	write_imageui(us1_line_counts, 0, 4);
	for(int i = 0; i < 4; ++i)
	{
		write_imagei(ic2_line_data, test_points[i], (int4)(test_points[1+i] - test_points[i], 0, -1));
	}
}