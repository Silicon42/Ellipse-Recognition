#ifndef CAST_HELPERS_CL
#define CAST_HELPERS_CL
//macro for splitting the lower half of a uint into an int2 with the low byte in the x and the high byte in the y
#define SPLIT_INDEX(uindex)	(convert_int2(((union s_conv)(uindex)).uc))
//TODO: check if unions are faster or pointer conversion macros
#define AS_LONG(x)	(*(long*)&(x))

union f_i_conv{
	float f;
	int i;
};

union f2_i2_conv{
	float2 f;
	int2 i;
};

union ui4_array{
	uint4 ui4;
	uint arr[4];
};

union l_conv{
	long l;
	int2 i;
	uint2 ui;
	char8 c;
	uchar8 uc;
	uchar uca[8];
};

union i_conv{
	int i;
	char4 c;
	uchar4 uc;
	uchar uca[4];
};

union s_conv{
	short s;
	ushort us;
	uint ui;
	char2 c;
	uchar2 uc;
	char ca[2];
};

#endif//CAST_HELPERS_CL