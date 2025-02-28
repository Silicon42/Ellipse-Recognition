// simple kernel for visualizing if the solver code works and is a significant quality improvement over the fast approximation
// given enough points, only draws lines from arc starts to calculated foci for items with 5+ points

kernel void conic_solve_debug(
	read_only image2d_t ff4_pseudo_coeffs,
	write_only image2d_t uc4_out)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	float16 coeffs = 0;
	coeffs.hi.lo = read_imagef(ff4_pseudo_coeffs, (int2)(coords.x*4+2, coords.y));
	if(coeffs.s8 < 5)	// skip drawing if less than 5 points involved
		return;
	coeffs.hi.hi = read_imagef(ff4_pseudo_coeffs, (int2)(coords.x*4+3, coords.y));
	coeffs.lo.hi = read_imagef(ff4_pseudo_coeffs, (int2)(coords.x*4+1, coords.y));
	coeffs.lo.lo = read_imagef(ff4_pseudo_coeffs, (int2)(coords.x*4, coords.y));

	

}