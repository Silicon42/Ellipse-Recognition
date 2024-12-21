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
int getStringIndex(char** list, const char* str);

// reads in a list of files by the names of the kernel functions, and compiles and links them into one cl_program object which it returns
// also populates the kprogs list with compiled program objects from the individual source files.
cl_program buildKernelProgsFromSource(cl_context context, cl_device_id device, const char* src_dir, QStaging* staging, const char* args, cl_program* kprogs, clbp_Error* e);
// takes a NULL terminated array of KernStaging pointers and an array of kernels and fills in the QStage array and argTracker array
// according to their details
void prepQStages(cl_context context, const QStaging* staging, const cl_program kprog, QStage* stages, ArgTracker* at, clbp_Error* e);
// creates a single channel cl_mem image from a file and attaches it to the tracked arg pointer provided
// the tracked arg must have the format pre-populated with a suitable way to interpret the raw image data
cl_mem imageFromFile(cl_context context, char const* fname, cl_image_format const* format, clbp_Error* e);

// converts format of data read from device to char array suitable for writing to typical image file
unsigned char readImageAsCharArr(char* data, TrackedArg* arg);

//TODO: verify that this is the correct header to place this function in
// Returned pointer must be freed when done using
char* readFileToCstring(char* fname, clbp_Error* e);

#endif//CL_BOILERPLATE_H