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
// checks provided channel type against one requested via the metadata, while some
// mismatches could result in undefined behavior, others are just likely not what
// you intended if you specified the metadata correctly, such as requesting a read
// from a full range float but providing a signed normalized value instead. This is
// primarily just to warn you that you might have made a mistake, but if it was
// intentional, you can safely ignore the warning that will follow
char isMatchingChannelType(const char* metadata, cl_channel_type type);
// if the channel type has a restricted order, returns the order that best fits normal types
//inline enum clChannelOrder isChannelTypeRestrictedOrder(enum clChannelType const type);
// returns the difference in number of channels provided vs requested,
inline char ChannelOrderDiff(char ch_cnt_data, cl_channel_order order);

//cl_channel_type getTypeFromMetadata(const char* metadata);

cl_channel_order getOrderFromChannelCnt(uint8_t count);

// get the minimum per pixel allocation size for reading output buffers to the host
uint8_t getPixelSize(cl_image_format format);

// in can be NULL if mode is EXACT or SINGLE
// This is a massive oversimplification since NDRanges aren't capped at 3, but
// that's all I expect to ever need from this and it makes implementation much easier
// plus it's the minimum required upper limit for non-custom device types in the spec
// so it's the maximum reliably portable value
char calcSizeByMode(Size3D const* ref, RangeData const* range, Size3D* ret);

//TODO: remove/replace these functions, they're no longer is useful for their original purpose and are only used in a print
//char getDeviceRWType(cl_channel_type type);
//char getArgStorageType(cl_channel_type type);

// currently assumes number of channels is the only thing important, NOT posistioning or ordering
uint8_t getChannelCount(cl_channel_order order);
//uint8_t getChannelWidth(char metadata_type);


// this only issues warnings to the user since they could easily have misnamed it
// and it isn't required data unlike on writes that need new textures
//void verifyReadArgTypeMatch(cl_image_format ref_format, char* metadata);

// converts an end relative arg index to a pointer to the referenced TrackedArg with error checking
//TrackedArg* getRefArg(const ArgTracker* at, uint16_t rel_ref);

//void setKernelArgs(cl_context context, const KernStaging* stage, cl_kernel kernel, ArgTracker* at);


#endif//CLBP_UTILS_H