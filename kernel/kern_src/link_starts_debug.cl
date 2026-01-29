// displays left and right link direction of input via a 3x bigger output image
#include "link_macros.cl_h"
//#include "offsets_LUT.cl_h"

__kernel void link_starts_debug(
	read_only image2d_t uc1_cont,
	write_only image2d_t uc4_debug_image)
{
	constant const int2 offsets[] = {(int2)(1,0),1,(int2)(0,1),(int2)(-1,1),(int2)(-1,0),-1,(int2)(0,-1),(int2)(1,-1)};
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	uchar cont_data = read_imageui(uc1_cont, coords).x;
/*
if(all(coords == 0))
printf("%v2i	%v2i	%v2i	%v2i	%v2i	%v2i	%v2i	%v2i	\n",offsets[0],offsets[1],offsets[2],offsets[3],offsets[4],offsets[5],offsets[6],offsets[7]);
return;
*/	if(!cont_data)	// only process populated cells
		return;
	
	coords = coords * 3 + 1;
	uchar isStart = (cont_data & IS_START) ? -1:0;
	uchar isEndAdj = (cont_data & ISNT_END_ADJ) ? 0:-1;

	// default to red, yellow if start, magenta if end adjacent, white if both
	write_imageui(uc4_debug_image, coords, (uint4)(-1, isStart, isEndAdj,-1));
	write_imageui(uc4_debug_image, coords + offsets[cont_data & R_CONT_IDX_MASK], (uint4)(0,-1,0,-1));	// green for right pixel
}