#ifndef CAST_HELPERS_CL
#define CAST_HELPERS_CL

union ul2_conv{
	ulong2 ul;
	uint4 ui;
	uchar16 uc;
};

union l_conv{
	long l;
	int2 i;
	uint2 ui;
	char8 c;
	uchar8 uc;
	char a[8];
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
};

#endif//CAST_HELPERS_CL