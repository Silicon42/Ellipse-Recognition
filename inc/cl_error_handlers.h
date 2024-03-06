#ifndef CL_ERROR_HANDLERS_H
#define CL_ERROR_HANDLERS_H

#include <CL/cl.h>

void handleClError(cl_int cl_error, const char* from);
int handleClGetDeviceIDs(cl_int cl_error);
void handleClBuildProgram(cl_int cl_error, cl_program program, cl_device_id device);

#endif //CL_ERROR_HANDLERS_H
