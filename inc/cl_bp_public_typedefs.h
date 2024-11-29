#ifndef CL_BP_PUBLIC_TYPEDEFS_H
#define CL_BP_PUBLIC_TYPEDEFS_H
/**
 * Typedefs exposed for data-driven staged queue creation
*/
#include <CL/cl.h>
//#include <stdint.h>
#include <stdbool.h>

enum rangeMode {
//	PAD,	// pad to multiple of work group dimensions
	EXACT,	// set range/size to param
//	SINGLE,	// meant for primarily serial workloads, [0] == false -> 1 hardware workgroup, [0] == true -> single work item
	REL,	// expand/contract range and output relative to input
	DIAG,	// exact on [0], contraction(- only, + no useful effect) relative to length of diagonal on [1]*, relative on [2], used for hough_lines
	DIVIDE,	// divides each component by corresponding param
	MULT,	// multiplies each component by corresponding param
	ROW,	// REL on y axis, EXACT on x and z
	COLUMN,	// REL on x axis, EXACT on y and z
};

typedef struct {
	int param[3];			// effects execution range and size of the output buffers, see rangeMode above
	int ref_idx;			// index of the arg whose size is the reference size that any relative size calculations will be based on
	enum rangeMode mode;	// what mode to calculate the NDRange/size_t[3] in
} RangeData;

typedef struct {
	RangeData size;			// data on how to calculate the size_t[3] of the arg
	bool is_host_readable;	// indicates whether to forcibly make an arg readable by the host, otherwise default is false unless it's the output of the last kernel
	char type;				// indicates what broad type of argument this should be, 'i'mage, image 'a'rray, 'p'ipe, 'b'uffer, or 's'calar
	char storage_type;		// the C type that the arg contents behave as	//TODO: pick char values for each type
} ArgStaging;	//TODO: since stbi only supports 8 bit depth the host readable flag forces 8 bit output which may cause calculation issues if buffer isn't last

// user provided info of how to set up kernels in a queue and their arguments
typedef struct {
	int kernel_idx;		// index of the reference kernel provided for cloning
	RangeData range;	// data on how to calculate the NDRange
	int* arg_idxs;		// array containing indices for each arg to use, freed when using freeStagingArray()	//TODO: write freeStagingArray()
} QStaging;				//TODO:^this should probably be separated out so that range isn't tied to buffer size

// info used in assigning an arg to kernels, creating/reading buffers on the host, and deallocating mem objects
typedef struct {
	cl_mem arg;				// pointer to the image or generic cl_mem object in question
	size_t size[3];			// used for calculating later arg sizes and host buffer sizes
	cl_image_format format;	// used for verifying compatible channel types, spacing and read/write operations
} TrackedArg;

typedef struct {
	TrackedArg* args;
	uint16_t args_cnt;
	uint16_t max_args;
	size_t max_out_size;
} ArgTracker;

// info actually used in enqueueing kernels
typedef struct {
	cl_kernel kernel;	// stage's kernel pointer gets stored here
	size_t range[3];	// stage's range size gets stored here
	char name[32];			// name of the kernel function, only used for user convenience/debugging
} QStage;

#endif//CL_BP_PUBLIC_TYPEDEFS_H