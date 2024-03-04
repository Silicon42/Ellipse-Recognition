#ifndef COMMON_ERROR_HANDLERS_H
#define COMMON_ERROR_HANDLERS_H

#include <CL/cl.h>

void handleClError(cl_int cl_error, const char* from);
void handleClGetDeviceIDs(cl_int cl_error);

#endif //COMMON_ERROR_HANDLERS_H
