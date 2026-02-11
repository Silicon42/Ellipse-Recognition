/*
converts conics to foci major form for easier graphing
*/
#include "conic_solve.cl_h"
#include "arc_data.cl_h"

kernel void to_foci_major(
	read_only image2d_t ff4_abcd,
	read_only image2d_t ff1_e,
	write_only image2d_t ff4_foci,
	write_only image2d_t ff1_major)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	Conic conic = {.fm = {
		.foci = read_imagef(ff4_abcd, coords),
		.major = read_imagef(ff1_e, coords).x
	}};

	if(all(conic.fm.foci == 0))
		return;

printf("v");
	convertConicGeneralToFociMajor(&conic, true);

	write_imagef(ff4_foci, coords, conic.fm.foci);
	write_imagef(ff1_major, coords, conic.fm.major);
}