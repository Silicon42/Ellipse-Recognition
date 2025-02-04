#ifndef CL_DEBUGABLE_TYPES_H
#define CL_DEBUGABLE_TYPES_H
// enum and bitfield equivalents for the cl.h defined types so they can actually be debugged
//NOTE: not all types are represented here this is WIP
#define CLBP_OFFSET_CHANNEL_ORDER	0x10B0
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
	//RESERVED/UNKNOWN	= 0x10F8 thru 0x10FF
	CLBP_INVALID_MEM_TYPE
};

enum clChannelOrder {
	CLBP_R			= 0x10B0,
	CLBP_A			= 0x10B1,
	CLBP_RG			= 0x10B2,
	CLBP_RA			= 0x10B3,
	CLBP_RGB		= 0x10B4,
	CLBP_RGBA		= 0x10B5,
	CLBP_BGRA		= 0x10B6,
	CLBP_ARGB		= 0x10B7,
	CLBP_INTENSITY	= 0x10B8,
	CLBP_LUMINANCE	= 0x10B9,
	CLBP_Rx			= 0x10BA,
	CLBP_RGx		= 0x10BB,
	CLBP_RGBx		= 0x10BC,
	CLBP_DEPTH		= 0x10BD,
	CLBP_sRGB		= 0x10BF,
	CLBP_sRGBx		= 0x10C0,
	CLBP_sRGBA		= 0x10C1,
	CLBP_sBGRA		= 0x10C2,
	CLBP_ABGR		= 0x10C3,
//RESERVED/UNKNOWN	= 0x10C4 thru 0x10CF
	CLBP_INVALID_CHANNEL_ORDER
};

#endif//CL_DEBUGABLE_TYPES_H