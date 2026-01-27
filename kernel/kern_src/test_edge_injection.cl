// injects a series of 5 points in place of the line_segments kernel for testing the arc_builder code for correctness

kernel void test_edge_injection(
	write_only image1d_t is2_start_coords,
	write_only image2d_t ic2_line_data,
	write_only image1d_t us1_line_counts)
{
	//this set of test points should result in an ellipse with foci (123,83) and (37,-3), abs center (80,40), 1st point rel center (60,20)
	// and eqn: -x^2 + xy - y^2 + 120x = 2000 for absolute coords or -x^2 + xy - y^2 + 100x - 20y = 0 for 1st point relative coords
//	int2 test_points[5] = {20, (int2)(40,60), (int2)(60,80), (int2)(100,100), (int2)(120,100)};

	//this set of test points should result in an ellipse with foci (28.0599915,-8.8354492) and (5.9400108,4.8354485), abs center (17,-2)
	//1st point rel foci (26.0599915,-12.8354492) and (3.9400108,0.8354485), rel center (15,-6), major axis length 33.077084
//	int2 test_points[5] = {(int2)(2,4), (int2)(4,8), (int2)(8,10), (int2)(14,10), (int2)(20,8)};

	//this set of test points should result in an ellipse with general form -2.609113x^2 + xy -4.380575y^2 -4.795691x +1572.440370y = 0
	// foci (190.835175,224.536957) and (-122.386864,142.233261), center (34.22415, 183.385116), major axis length 482.317895
//	int2 test_points[5] = {0, (int2)(42,3), (int2)(85, 12), (int2)(127,27), (int2)(173,52)};	//rel
	// when calculated through matrix inversion, due to rounding differences, this results in an ellipse with absolute
	// foci (360.975220,237.306137) and (229.548965,495.015808), major axis length 744.057678
	int2 test_points[5] = {(int2)(321,0), (int2)(363,3), (int2)(406, 12), (int2)(448,27), (int2)(494,52)};	//abs

	//this set of test points should result in an hyperbola with foci (4*sqrt(2), 4*sqrt(2)) and (-4*sqrt(2), -4*sqrt(2)) abs center (0,0)
	//1st point rel center (-16,-1), major axis length 8*sqrt(2) = 11.3137085
//	int2 test_points[5] = {(int2)(16,1), (int2)(8,2), (int2)(4,4), (int2)(2,8), (int2)(1,16)};

	//this set of test points should result in an hyperbola with foci (3.747314, -1.348730) and (5.252686, 3.348731) abs center (4.5, 1)
	//1st point rel foci (3.747314, -9.348730) and (5.252686, -4.651269), rel center (4.5, -7), major axis length 3.055744
	//-5x^2 + 10xy + 9y^2 + 35x - 63y = 72, rel -5x^2 + 10xy + 9y^2 + 115x + 81y = 0
//	int2 test_points[5] = {(int2)(0,8), (int2)(3,4), (int2)(4,3), (int2)(9,3), (int2)(12,4)};
	//-5x^2 - 10xy + 9y^2 + 115x - 81y = 0
//	int2 test_points[5] = {(int2)(0,0), (int2)(3,4), (int2)(4,5), (int2)(9,5), (int2)(12,4)};

	write_imagei(is2_start_coords, 0, (int4)(test_points[0], 0, -1));
	write_imageui(us1_line_counts, 0, 4);
	for(int i = 0; i < 4; ++i)
	{
		write_imagei(ic2_line_data, test_points[i], (int4)(test_points[1+i] - test_points[i], 0, -1));
	}
}