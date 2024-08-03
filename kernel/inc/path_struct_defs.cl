#ifndef PATH_STRUCT_DEFS_CL
#define PATH_STRUCT_DEFS_CL

/*
The path accumulator is arrange as a packed series of 3-bit chunks representing 
 offset indices and a 6-bit length stored across 2 longs and read and written to
 the image buffer as a uint4 such that the contents fit in a single read/write 
 operation. The layout is as follows:

	|64		|32		|24		|16		|8		|0
hi	 x[N][M][L][K][J][I][H][G][F][E][D][C][B]
lo	 x[A][9][8][7][6][5][4][3][2][1][0][ LEN]

where x denotes an unused bit.
 LEN has valid values 1 thru 40 with values > 40 indicating that it stopped due 
 to running out of space as opposed to angular acceleration
*/

// defines for how many octals may be packed into path accumulator longs
#define ACCUM_STRUCT_LEN1 19
#define ACCUM_STRUCT_LEN2 40
#define LEN_BITS_MASK 0x3F
/*
union ul2_conv{
	ulong2 ul;
	uint4 ui;
};
*/
void write_data_accum(ulong2 accum, uchar len, write_only image2d_t ui4_path_image, int2 coords)
{
	// lengths greater than 19 required an accumulator swap, so swap back
	if(len > ACCUM_STRUCT_LEN1)
		accum = accum.yx;

//	printf("%2i %11o %10o %11o %10o\n", len, accum.y >> 30, accum.y & 0x1FFFFFFF, accum.x >> 30, accum.x & 0x1FFFFFFF);

	accum.x = (accum.x << 6) | len;

	write_imageui(ui4_path_image, coords, *(uint4*)&accum);
}

uchar read_data_accum(ulong2* accum, read_only image2d_t ui4_path_image, int2 coords)
{
	uint4 packed = read_imageui(ui4_path_image, coords);

	uchar len = packed.x & LEN_BITS_MASK;
	*(uint4*)accum = packed;
	//printf("%o %o\n", (*accum).x, (*accum).y);
	(*accum).x >>= 6;
//	printf("%2i %o%10o%10o %9o%10o\n", len, (*accum).y >> 60, ((*accum).y >> 30) & 0x3FFFFFFF, (*accum).y & 0x3FFFFFFF, (*accum).x >> 30, (*accum).x & 0x3FFFFFFF);

	return len;
}

#endif//PATH_STRUCT_DEFS_CL