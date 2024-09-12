#ifndef CAST_HELPERS_CL
#define CAST_HELPERS_CL

//TODO: check if unions are faster or pointer conversion macros
#define AS_LONG(x)	(*(long*)&(x))

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
	char2 c;
	uchar2 uc;
	char ca[2];
};

#endif//CAST_HELPERS_CL