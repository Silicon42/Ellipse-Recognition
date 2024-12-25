#ifndef CLBP_UTILS_H
#define CLBP_UTILS_H
/**
 * This file contains helper functions to cl_boilerplate.c, and is not neccessarily
 * generally useful if you just want to use the boilerplate to create and run a 
 * staged queue of image operations, it's provided publicly in the off chance that
 * you do end up needing it though.
 */

//#define CL_VERSION_2_0
//#define CL_VERSION_2_1
#include <CL/cl.h>
#include "clbp_public_typedefs.h"

cl_mem createImageBuffer(cl_context context, char force_host_readable, char is_array, const cl_image_format* img_format, const size_t img_size[3]);
// validates metadata[0 thru 2] formating and returns true if valid
char isArgMetadataValid(char const metadata[static 3]);
cl_channel_type getTypeFromMetadata(const char* metadata);
cl_channel_order getOrderFromMetadata(const char* metadata);

// in can be NULL if mode is EXACT or SINGLE
char calcSizeByMode(Size3D const* ref, RangeData const* range, Size3D* ret);

char getDeviceRWType(cl_channel_type type);
char getArgStorageType(cl_channel_type type);

// currently assumes number of channels is the only thing important, NOT posistioning or ordering
unsigned char getChannelCount(cl_channel_order order);
unsigned char getChannelWidth(char metadata_type);


// this only issues warnings to the user since they could easily have misnamed it
// and it isn't required data unlike on writes that need new textures
void verifyReadArgTypeMatch(cl_image_format ref_format, char* metadata);

// converts an end relative arg index to a pointer to the referenced TrackedArg with error checking
//TrackedArg* getRefArg(const ArgTracker* at, uint16_t rel_ref);

void setKernelArgs(cl_context context, const KernStaging* stage, cl_kernel kernel, ArgTracker* at);


#endif//CLBP_UTILS_H