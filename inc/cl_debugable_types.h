#ifndef CL_DEBUGABLE_TYPES_H
#define CL_DEBUGABLE_TYPES_H
// enum and bitfield equivalents for the cl.h defined types so they can actually be debugged
//NOTE: not all types are represented here this is WIP
#define CLBP_OFFSET_CHANNEL_TYPE	0x10D0
enum clChannelType {
	CLBP_SNORM_INT8			= 0x10D0,
	CLBP_SNORM_INT16		= 0x10D1,
	CLBP_UNORM_INT8			= 0x10D2,
	CLBP_UNORM_INT16		= 0x10D3,
	CLBP_UNORM_SHORT_565	= 0x10D4,
	CLBP_UNORM_SHORT_555	= 0x10D5,
	CLBP_UNORM_INT_101010	= 0x10D6,
	CLBP_SIGNED_INT8		= 0x10D7,
	CLBP_SIGNED_INT16		= 0x10D8,
	CLBP_SIGNED_INT32		= 0x10D9,
	CLBP_UNSIGNED_INT8		= 0x10DA,
	CLBP_UNSIGNED_INT16		= 0x10DB,
	CLBP_UNSIGNED_INT32		= 0x10DC,
	CLBP_HALF_FLOAT			= 0x10DD,
	CLBP_FLOAT				= 0x10DE,
	//RESERVED/UNKNOWN		= 0x10DF,
	CLBP_UNORM_INT_101010_2	= 0x10E0,
	//RESERVED/UNKNOWN		= 0x10E1 thru 0x10EF
	CLBP_INVALID_CHANNEL_TYPE
};

#define CLBP_OFFSET_MEMTYPE 0x10F0
enum clMemType {
	CLBP_BUFFER			= 0x10F0,
	CLBP_IMAGE2D		= 0x10F1,
	CLBP_IMAGE3D		= 0x10F2,
	CLBP_IMAGE2D_ARRAY	= 0x10F3,
	CLBP_IMAGE1D		= 0x10F4,
	CLBP_IMAGE1D_ARRAY	= 0x10F5,
	CLBP_PIPE			= 0x10F6,
	CLBP_IMAGE1D_BUFFER	= 0x10F7,
//	RESERVED/UNKNOWN	= 0x10F8 thru 0x10FF
	CLBP_INVALID_MEM_TYPE
};

#define CLBP_OFFSET_CHANNEL_ORDER	0x10B0
enum clChannelOrder {			// is min support	| clamped color alpha channel behavior
	CLBP_R			= 0x10B0,	// all, R&W			| 1
	CLBP_A			= 0x10B1,	// n				| 0
	CLBP_RG			= 0x10B2,	// 2.x				| 1
	CLBP_RA			= 0x10B3,	// n				| 0
	CLBP_RGB		= 0x10B4,	// n				| 1
	CLBP_RGBA		= 0x10B5,	// all, R&W			| 0
	CLBP_BGRA		= 0x10B6,	// all, unorm8		| 0
	CLBP_ARGB		= 0x10B7,	// n				| 0
	CLBP_INTENSITY	= 0x10B8,	// n				| 0
	CLBP_LUMINANCE	= 0x10B9,	// n				| 1
	CLBP_Rx			= 0x10BA,	// n				| 0
	CLBP_RGx		= 0x10BB,	// n				| 0
	CLBP_RGBx		= 0x10BC,	// n				| 0
	CLBP_DEPTH		= 0x10BD,	// 2.x				| ?
	CLBP_sRGB		= 0x10BF,	// n				| 1?
	CLBP_sRGBx		= 0x10C0,	// n				| 0?
	CLBP_sRGBA		= 0x10C1,	// 2.x, unorm8		| 0?
	CLBP_sBGRA		= 0x10C2,	// n				| 0?
	CLBP_ABGR		= 0x10C3,	// n				| 0?
//RESERVED/UNKNOWN	= 0x10C4 thru 0x10CF
	CLBP_INVALID_CHANNEL_ORDER
};

#endif//CL_DEBUGABLE_TYPES_H