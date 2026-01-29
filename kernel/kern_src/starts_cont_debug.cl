// compares continuation data from uc1_starts_cont to uc1_cont for sanity check purposes
#include "link_macros.cl_h"

kernel void starts_cont_debug(
	read_only image2d_t uc1_cont,
	read_only image2d_t uc1_starts_cont,
	write_only image2d_t uc4_retrace)
{
	const int2 coords = (int2)(get_global_id(0), get_global_id(1));

	uchar cont_data = read_imageui(uc1_starts_cont, coords).x;

	if(!cont_data)
		return;

	cont_data ^= read_imageui(uc1_cont, coords).x;
	cont_data &= R_CONT_IDX_MASK;
	if(cont_data)
		printf("%v2i starts cont mismatch\n", coords);
}
