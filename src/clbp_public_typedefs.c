#include "clbp_public_typedefs.h"

char const* modeNames[] = {
	"ADD_SUB",
	"MULTIPLY",
	"DIVIDE"
	"EXACT",
	"CLBP_RM_ROW",
	"CLBP_RM_COLUMN",
	"DIAGONAL",
	//
	NULL
};

char const* channelTypes[] = {
	"SNORM_INT8",
	"SNORM_INT16",
	"UNORM_INT8",
	"UNORM_INT16",
	"UNORM_SHORT_565",
	"UNORM_SHORT_555",
	"UNORM_INT_101010",
	"SIGNED_INT8",
	"SIGNED_INT16",
	"SIGNED_INT32",
	"UNSIGNED_INT8",
	"UNSIGNED_INT16",
	"UNSIGNED_INT32",
	"HALF_FLOAT",
	"FLOAT",
	"",	// RESERVED/UNKOWN
	"UNORM_INT_101010_2",
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
