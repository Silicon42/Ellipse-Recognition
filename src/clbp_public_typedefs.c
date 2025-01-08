#include "clbp_public_typedefs.h"

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
	"IMAGE2D",
	"IMAGE3D",
	"IMAGE2D_ARRAY",
	"IMAGE1D",
	"IMAGE1D_ARRAY",
	"IMAGE1D_BUFFER",
	"PIPE",
	NULL
};
