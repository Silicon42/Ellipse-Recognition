#ifndef CL_BOILERPLATE_H
#define CL_BOILERPLATE_H
/**
 * This file contains the function declarations that someone who just wants to
 * create a simple data-driven staged queue would need
*/

//#define CL_VERSION_2_0
//#define CL_VERSION_2_1
#include <CL/cl.h>
#include "cl_bp_public_typedefs.h"


// attempts to get the first available GPU or if none available CPU
//TODO: actually implement multiple attempts to find a GPU, currently just takes the first device of the first platform
cl_device_id getPreferredDevice();

// adds the char* to the char* array if the contents are unique, the char* array
// MUST have unused entries filled with null pointers with an additional null
// pointer at list[max_entries]
// returns -1 if entry already exists, 0 if out of space, and 1 if entry was unique
int addUniqueString(const char** list, int max_entries, const char* str);

// reads in a list of files by the names of the kernel functions, builds them and makes a kernel for each in an array up to 
// max_kernels, returns the number of kernels actually created
cl_uint buildKernelsFromSource(cl_context context, cl_device_id device, const char* src_dir, const char** names, const char* args, cl_kernel* kernels, cl_uint max_kernels);
// takes a NULL terminated array of QStaging pointers and an array of kernels and fills in the QStage array and argTracker array
// according to their details
int prepQStages(cl_context context, const QStaging** staging, const cl_kernel* ref_kernels, QStage* stages, int max_stages, ArgTracker* at);
// creates a single channel cl_mem image from a file and attaches it to the tracked arg pointer provided
// the tracked arg must have the format pre-populated with a suitable way to interpret the raw image data
void imageFromFile(cl_context context, const char* fname, TrackedArg* tracked);

// converts format of data read from device to char array suitable for writing to typical image file
unsigned char readImageAsCharArr(char* data, TrackedArg* arg);

// Returned value must be freed when done using
char* readFileToCstring(const char* fname);

#endif//CL_BOILERPLATE_H