#include "clbp_public_typedefs.h"

char const* modeNames[] = {
	"ADD_SUB",
	"MULTIPLY",
	"DIVIDE",
	"EXACT",
	"CLBP_RM_ROW",
	"CLBP_RM_COLUMN",
	"DIAGONAL",
	//
	NULL
};

char const* channelTypes[] = {
	"snorm8",
	"snorm16",
	"unorm8",
	"unorm16",
	"unorm5-6-5",
	"unorm5-5-5",
	"unorm10-10-10",
	"int8",
	"int16",
	"int32",
	"uint8",
	"uint16",
	"uint32",
	"half",
	"float",
	"",	// RESERVED/UNKOWN
	"unorm10-10-10-2",
	NULL	// End of spec defined types, others may exist
};

char const* memTypes[] = {
	"BUFFER",
	"image2d_t",
	"image3d_t",
	"IMAGE2D_ARRAY",
	"image1d_t",
	"IMAGE1D_ARRAY",
	"IMAGE1D_BUFFER",
	"PIPE",
	NULL
};
