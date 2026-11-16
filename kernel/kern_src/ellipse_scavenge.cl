// re-calculates the ellipse equation using all the arcs that got grouped together for stability purposes
// then searches for any remaining single segments that can be attributed to the solution and recalculates
// a final ellipse solution
// for high precision use cases, could be extended to iterating through all edge pixels instead of just line
// segment endpoints


#include "cast_helpers.cl_h"
#include "conic_solve.cl_h"

kernel void ellipse_scavenge(
	read_only image2d_t ii4_group_list,
	read_only image2d_t is2_arc_coords,
	read_only image2d_t ff4_pre_solve_coeffs,
	read_only image2d_t ic2_line_data,
	write_only image2d_t ff4_ellipse_ACDE,
	write_only image2d_t ff1_ellipse_B)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	union s16_conv arcs;
	arcs.i.lo = read_imagei(ii4_group_list, indices);
	if(arcs.a[7] >= 0)
		arcs.i.hi = read_imagei(ii4_group_list, indices | (int2)(0,2));
	
	// sum the pre-solve coeffs for all associated arcs of the ellipse
	float16 curr_coeffs, coeff_accum = 0;
	for(int i = 0; i < 16; ++i)
	{
		// break out if all valid entries in list are done
		if(arcs.a[i] < 0)
			break;
		
		int2 arc_coords = read_imagei(is2_arc_coords, indices).lo;
		readPreSolveCoeffs(ff4_pre_solve_coeffs, arc_coords, &curr_coeffs);
		coeff_accum += curr_coeffs;
	}

	// re-solve for the ellipse for numeric stability purposes now that we have all associated arcs accounted for
	Conic ellipse;
	solveConic((union f16_conv){.v = coeff_accum}.a, ellipse.general);

//TODO: add the scavenging portion, this skips that for now for debugging the dedup phase
	write_imagef(ff4_ellipse_ACDE, indices, ellipse.fm.foci);
	write_imagef(ff1_ellipse_B, indices, ellipse.fm.major);
}