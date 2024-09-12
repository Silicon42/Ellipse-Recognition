#ifndef PATH_STRUCT_DEFS_CL
#define PATH_STRUCT_DEFS_CL

#include "offsets_LUT.cl"
#include "math_helpers.cl"

//TODO: actually define this as a bitfield
/*
The path accumulator is arrange as a packed series of 3-bit chunks representing 
 offset indices and a 6-bit length stored across 2 longs and read and written to
 the image buffer as a uint4 such that the contents fit in a single read/write 
 operation. The layout is as follows:

	|64		|56		|48		|40		|32		|24		|16		|8		|0
lo	 _[K][J][I][H][G][F][E][D][C][B][A][9][8][7][6][5][4][3][2][1][0]
	|							<--y|							<--x|
hi	 _[d][c][b][a][Z][Y][X][W][V][U][T][S][R][Q][P][O][N][M][L][ LEN]
	|							<--w|							<--z|
where _ denotes an unused bit.
 LEN has valid values 1 thru 40 with values > 40 indicating that it stopped due 
 to running out of space as opposed to angular acceleration
*/

// defines for how many octals may be packed into path accumulator longs
#define ACCUM_STRUCT_LEN1 21
#define ACCUM_STRUCT_LEN2 40
#define LEN_BITS_MASK 0x3F

// arcs consisting of less than this many pixels get flaged as short
#define SHORT_ARC_THRESH 5

struct arc_data{	//TODO: check how to ensure optimal packing
	char is_flat;		// flag for if the arc segment has little deflection on it's length
	char is_short;		// flag for if the arc's length consists of < SHORT_ARC_THRESH pixels
	char ccw_mult;		// represents arc's handedness, ie if the gradient is inward or outward, 1 for cw, -1 for ccw
	char2 offset_end;	// position delta of the end coords from the start coords
	char2 offset_mid;	// position delta of the arc midpoint coords from the start coords
	float2 center;		// rough estimate of the center coords of the arc
};

// used only for reading/writing arc_data to an image buffer
union arc_rw{
	struct arc_data data;
	uint4 ui4;
	ulong2 ul2;
};

void write_data_accum(ulong2 accum, char len, write_only image2d_t ui4_path, write_only image2d_t ui4_arc_data, const int2 base_coords, const int2 end_coords)
{
	// lengths less than or equal 21 haven't had an accumulator swap, so swap now to maintain correct order
	if(len <= ACCUM_STRUCT_LEN1)
		accum = accum.yx;

	//printf("%2i %11o %10o %11o %10o\n", len, accum.y >> 30, accum.y & 0x1FFFFFFF, accum.x >> 30, accum.x & 0x1FFFFFFF);

	accum.y |= len;

	write_imageui(ui4_path, base_coords, *(uint4*)&accum);

	// set arc data in struct for writing
	union arc_rw arcRW;
	struct arc_data* arc = &(arcRW.data);
	arc->is_short = len < SHORT_ARC_THRESH;
	// get end offset vector
	int2 offset_end = end_coords - base_coords;
	arc->offset_end = convert_char2(offset_end);

	// find the offset of the halfway point on the path by partially reconstructing it
	int2 offset_mid = 0;
	for(int i = len / 2; i > 0; --i)
	{
		offset_mid += offsets[accum.x & 7];
		accum.x >>= 3;
	}

	arc->offset_mid = convert_char2(offset_mid);

	// determine state of flag for flatness which we define as the midpoint of the chord having a small displacement
	// from the approximate halfway point of the arc, in the case of a nearly flat arc the halfway point should be 
	// nearly exactly the arc midpoint and very near to the chord midpoint
	int2 mid_displacement = 2*offset_mid - offset_end;
	mid_displacement *= mid_displacement;
	int disp_dist2 = mid_displacement.x + mid_displacement.y;
	arc->is_flat = disp_dist2 < 4;

	if(len < 2)
	{
		// calculations on this arc are worthless as it's too short to give good values
		// however since processing happens per connected edge, it can't be completely rejected outright
		arc->center = (float2)(NAN, NAN);
	}
	else
	{
		int scale_div = 2 * cross_2d_i(offset_end, offset_mid);
		if(scale_div == 0)	//prevent divide by zero
			arc->center = (float2)(INFINITY, INFINITY);
			// cw/ccw is meaningless here since it's flat and the start, mid and end points are co-linear
		else
		{
			//solve for center of circle
			int2 temp, perp_end, perp_mid;
			perp_mid = perp_2d_i(offset_mid);
			perp_end = perp_2d_i(offset_end);

			int mag2_end, mag2_mid;
			mag2_end = mag2_2d_i(offset_end);
			mag2_mid = mag2_2d_i(offset_mid);

			// center direction relative to start, not to scale yet
			float2 center = convert_float2(mag2_mid * perp_end - mag2_end * perp_mid);

			// cw arcs have center in similar direction to the cw perpendicular of the offset to the arc half point and
			// ccw arcs have center in opposing direction. This means cw arcs have positive dot product and ccw negative
			arc->ccw_mult = sign(dot(center, convert_float2(perp_mid)));
			center = convert_float2(base_coords) + (center / scale_div);
			arc->center = center;
		}
	}

	write_imageui(ui4_arc_data, base_coords, arcRW.ui4);
}

uchar read_data_accum(ulong2* accum, read_only image2d_t ui4_path, int2 coords)
{
	uint4 packed = read_imageui(ui4_path, coords);

	uchar len = packed.z & LEN_BITS_MASK;
	*(uint4*)accum = packed;
	//printf("%o %o\n", (*accum).x, (*accum).y);
	//(*accum).x >>= 6;
//	printf("%2i %o%10o%10o %9o%10o\n", len, (*accum).y >> 60, ((*accum).y >> 30) & 0x3FFFFFFF, (*accum).y & 0x3FFFFFFF, (*accum).x >> 30, (*accum).x & 0x3FFFFFFF);

	return len;
}

#endif//PATH_STRUCT_DEFS_CL