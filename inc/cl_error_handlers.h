#ifndef CL_ERROR_HANDLERS_H
#define CL_ERROR_HANDLERS_H

#include <CL/cl.h>

void handleClError(cl_int cl_error, const char* from);
void handleClGetDeviceIDs(cl_int cl_error);

#endif //CL_ERROR_HANDLERS_H
