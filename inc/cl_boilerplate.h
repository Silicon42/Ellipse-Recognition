#ifndef CL_BOILERPLATE_H
#define CL_BOILERPLATE_H
/**
 * This file contains the function declarations that someone who just wants to
 * create a simple data-driven staged queue would need
*/

//#define CL_VERSION_2_0
//#define CL_VERSION_2_1
#include <CL/cl.h>
#include "clbp_public_typedefs.h"


// attempts to get the first available GPU or if none available CPU
//TODO: actually implement multiple attempts to find a GPU, currently just takes the first device of the first platform
cl_device_id getPreferredDevice();

// adds the char* to the char* array if the contents are unique, the char* array
// MUST have unused entries filled with null pointers with an additional null
// pointer at list[max_entries]
// returns -1 if entry already exists, 0 if out of space, and 1 if entry was unique
//NOTE: does not check str for a null pointer
int addUniqueString(char** list, int max_entries, char* str);
// Searches through a string array (char** list) for a match to the contents of char* str
// Stops searching if it reaches a null pointer in the list. Returns -1 if no match is found.
//NOTE: does not check str for a null pointer
int getStringIndex(char const** list, char const* str);

// Initializes a StagedQ object's arrays and counts
cl_int allocStagedQArrays(QStaging const* staging, StagedQ* staged);

// applies the relative calculations for all arg sizes starting from the first non-hardcoded input argument
void calcRanges(QStaging const* staging, StagedQ* staged, clbp_Error* e);

// handles using staging data to selectively open kernel program source files and compile and link them into a single program binary
//TODO: add support for using pre-calculated ranges as defined constants
cl_program buildKernelProgsFromSource(cl_context context, cl_device_id device, const char* src_dir, QStaging* staging, const char* args, clbp_Error* e);

// creates actual kernel instances from staging data and stores it in the staged queue
void instantiateKernels(QStaging const* staging, const cl_program kprog, StagedQ* staged, clbp_Error* e);

// infers the access qualifiers of the image args as well as verifies that type data specified matches what the kernels expect of it
// meant to be run once after kernels have been instantiated for at least 1 staged queue, additional staged queues don't
// require re-runs of inferArgAccessAndVerifyFormats() since data extracted from the kernel instance args shouldn't change
void inferArgAccessAndVerifyFormats(QStaging* staging, StagedQ const* staged);

// fills in the ArgTracker according to the arg staging data in staging,
// assumes the ArgTracker was allocated big enough not to overrun it and
// is pre-populated with the expected number of hard-coded input entries
// such that it may add the first new entry at input_img_cnt
size_t instantiateImgArgs(cl_context context, QStaging const* staging, StagedQ* staged, clbp_Error* e);

// --returns the max number of bytes needed for reading out of any of the host readable buffers-- << not true anymore but might add back later
//TODO: add support for returning a list of host readable buffers
void setKernelArgs(QStaging const* staging, StagedQ* staged, clbp_Error* e);

// takes a NULL terminated array of KernStaging pointers and an array of kernels and fills in the QStage array and argTracker array
// according to their details
//void prepQStages(cl_context context, const QStaging* staging, const cl_program kprog, QStage* stages, ArgTracker* at, clbp_Error* e);
// creates a single channel cl_mem image from a file and attaches it to the tracked arg pointer provided
// the tracked arg must have the format pre-populated with a suitable way to interpret the raw image data
cl_mem imageFromFile(cl_context context, char const* fname, cl_image_format const* format, Size3D* size, clbp_Error* e);

// converts format of data to char array compatible read,
// data must point to a 32-bit aligned array. if it was malloc'd, it is aligned
// returns channel count since it's often needed after this and is already called here
uint8_t readImageAsCharArr(char* data, StagedQ const* staged, uint16_t idx);

//TODO: verify that this is the correct header to place this function in
// Returned pointer must be freed when done using
char* readFileToCstring(char* fname, clbp_Error* e);

void freeQStagingArrays(QStaging* staging);
void freeStagedQArrays(StagedQ* staged);


#endif//CL_BOILERPLATE_H