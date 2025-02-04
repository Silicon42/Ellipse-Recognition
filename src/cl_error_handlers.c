#include "cl_error_handlers.h"
#include <stdio.h>

const char undefinedErr[] = "20-29 CL error not in spec or error string list needs updating";

const char* clErrStr[] ={
/*  0*/ "CL_SUCCESS",
/* -1*/ "CL_DEVICE_NOT_FOUND",
/* -2*/ "CL_DEVICE_NOT_AVAILABLE",
/* -3*/ "CL_COMPILER_NOT_AVAILABLE",
/* -4*/ "CL_MEM_OBJECT_ALLOCATION_FAILURE",
/* -5*/ "CL_OUT_OF_RESOURCES",
/* -6*/ "CL_OUT_OF_HOST_MEMORY",
/* -7*/ "CL_PROFILING_INFO_NOT_AVAILABLE",
/* -8*/ "CL_MEM_COPY_OVERLAP",
/* -9*/ "CL_IMAGE_FORMAT_MISMATCH",
/*-10*/ "CL_IMAGE_FORMAT_NOT_SUPPORTED",
/*-11*/ "CL_BUILD_PROGRAM_FAILURE",
/*-12*/ "CL_MAP_FAILURE",
/*-13*/ "CL_MISALIGNED_SUB_BUFFER_OFFSET",
/*-14*/ "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
/*-15*/ "CL_COMPILE_PROGRAM_FAILURE",
/*-16*/ "CL_LINKER_NOT_AVAILABLE",
/*-17*/ "CL_LINK_PROGRAM_FAILURE",
/*-18*/ "CL_DEVICE_PARTITION_FAILED",
/*-19*/ "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
/*-20*/ undefinedErr,
/*-21*/ undefinedErr,
/*-22*/ undefinedErr,
/*-23*/ undefinedErr,
/*-24*/ undefinedErr,
/*-25*/ undefinedErr,
/*-26*/ undefinedErr,
/*-27*/ undefinedErr,
/*-28*/ undefinedErr,
/*-29*/ undefinedErr,
/*-30*/ "CL_INVALID_VALUE",
/*-31*/ "CL_INVALID_DEVICE_TYPE",
/*-32*/ "CL_INVALID_PLATFORM",
/*-33*/ "CL_INVALID_DEVICE",
/*-34*/ "CL_INVALID_CONTEXT",
/*-35*/ "CL_INVALID_QUEUE_PROPERTIES",
/*-36*/ "CL_INVALID_COMMAND_QUEUE",
/*-37*/ "CL_INVALID_HOST_PTR",
/*-38*/ "CL_INVALID_MEM_OBJECT",
/*-39*/ "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
/*-40*/ "CL_INVALID_IMAGE_SIZE",
/*-41*/ "CL_INVALID_SAMPLER",
/*-42*/ "CL_INVALID_BINARY",
/*-43*/ "CL_INVALID_BUILD_OPTIONS",
/*-44*/ "CL_INVALID_PROGRAM",
/*-45*/ "CL_INVALID_PROGRAM_EXECUTABLE",
/*-46*/ "CL_INVALID_KERNEL_NAME",
/*-47*/ "CL_INVALID_KERNEL_DEFINITION",
/*-48*/ "CL_INVALID_KERNEL",
/*-49*/ "CL_INVALID_ARG_INDEX",
/*-50*/ "CL_INVALID_ARG_VALUE",
/*-51*/ "CL_INVALID_ARG_SIZE",
/*-52*/ "CL_INVALID_KERNEL_ARGS",
/*-53*/ "CL_INVALID_WORK_DIMENSION",
/*-54*/ "CL_INVALID_WORK_GROUP_SIZE",
/*-55*/ "CL_INVALID_WORK_ITEM_SIZE",
/*-56*/ "CL_INVALID_GLOBAL_OFFSET",
/*-57*/ "CL_INVALID_EVENT_WAIT_LIST",
/*-58*/ "CL_INVALID_EVENT",
/*-59*/ "CL_INVALID_OPERATION",
/*-60*/ "CL_INVALID_GL_OBJECT",
/*-61*/ "CL_INVALID_BUFFER_SIZE",
/*-62*/ "CL_INVALID_MIP_LEVEL",
/*-63*/ "CL_INVALID_GLOBAL_WORK_SIZE",
/*-64*/ "CL_INVALID_PROPERTY",
/*-65*/ "CL_INVALID_IMAGE_DESCRIPTOR",
/*-66*/ "CL_INVALID_COMPILER_OPTIONS",
/*-67*/ "CL_INVALID_LINKER_OPTIONS",
/*-68*/ "CL_INVALID_DEVICE_PARTITION_COUNT",
/*-69*/ "CL_INVALID_PIPE_SIZE",
/*-70*/ "CL_INVALID_DEVICE_QUEUE",
/*-71*/ "CL_INVALID_SPEC_ID",
/*-72*/ "CL_MAX_SIZE_RESTRICTION_EXCEEDED"
};

void printClError(cl_int cl_error)
{
	if (cl_error >= -72 && cl_error <= 0)
		perror(clErrStr[-cl_error]);
	else
		perror(&undefinedErr[6]);	//slightly different output without numbers at beginning
}

//generic semi-verbose error handling for OpenCL functions
// cl_error:	error code provided by an OpenCL function
// from:		name of the function that generated the error code for user clarity
void handleClError(cl_int cl_error, const char* from)
{
	if(!cl_error)
		return;

	fputs(from, stderr);
	fputs("(): ", stderr);
	printClError(cl_error);

//	exit(cl_error);
}

int handleClGetDeviceIDs(cl_int cl_error)
{
	if(cl_error == CL_DEVICE_NOT_FOUND)
	{
		perror("No device of specified category on clGetDeviceIDs()\n");
		return 1;
	}
	//else
	handleClError(cl_error, "clGetDeviceIDs");
	return 0;
}

void printBuildLog(cl_program program, cl_device_id device)
{
	size_t log_size = 0;
	cl_int cl_error;

	// Find size of log and print to std output
	cl_error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	if(cl_error)
		handleClError(cl_error, "clGetProgramBuildInfo->logsize");
	printf("log_size: %zu\n", log_size);

	if(!log_size)
	{
		fputs("Might be multiple definition of functions/variables?\n", stderr);
		return;
	}

	char* program_log = (char*) malloc(log_size);
	if(program_log)
	{
		program_log[0] = '\0';
		cl_error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, program_log, NULL);
		if(cl_error)
			handleClError(cl_error, "clGetProgramBuildInfo->build log");
		
		//program_log[log_size] = '\0';
		fputs((const char*)program_log, stderr);
		free(program_log);
	}
	else
		fputs("Failed to allocate log.\n", stderr);
}

void handleClBuildProgram(cl_int cl_error, cl_program program, cl_device_id device)
{
	if(cl_error)
	{
		printClError(cl_error);
		fputs("------ BUILD FAILED ------\n", stderr);
		printBuildLog(program, device);
//		exit(1);
	}
}