#ifndef CLBP_PUBLIC_TYPEDEFS_H
#define CLBP_PUBLIC_TYPEDEFS_H
/**
 * Typedefs exposed for data-driven staged queue creation
*/
#include <CL/cl.h>
#include <stdint.h>
//#include <stdbool.h>
#include "cl_debugable_types.h"
#include "clbp_error_handling.h"

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

#define CLBP_CHANNELTYPE_OFFSET 0x10D0
char const* channelTypes[] = {
	"SNORM_INT8",
	"SNORM_INT16",
	"UNORM_INT8",
	"UNORM_INT16",
	"UNORM_SHORT_565",
	"UNORM_SHORT_555",
	"UNORM_INT_101010",
	"SIGNED_INT8",
	"SIGNED_INT16",
	"SIGNED_INT32",
	"UNSIGNED_INT8",
	"UNSIGNED_INT16",
	"UNSIGNED_INT32",
	"HALF_FLOAT",
	"FLOAT",
	"",	// RESERVED/UNKOWN
	"UNORM_INT_101010_2",
	NULL	// End of spec defined types, others may exist
};

#define CLBP_MEMTYPE_OFFSET 0x10F0
char const* memTypes[] = {
"BUFFER",
"IMAGE2D",
"IMAGE3D",
"IMAGE2D_ARRAY",
"IMAGE1D",
"IMAGE1D_ARRAY",
"IMAGE1D_BUFFER",
"PIPE",
};

typedef struct {
	size_t d[3];
} Size3D;
/*
typedef struct {
	uint8_t widthExp:3;		// width in bytes represented as 2^widthExp, Ex: int32_t would be 2
	uint8_t isUnsigned:1;	// whether the type is signed or not, ignored if isFloat is true
	uint8_t isFloat:1;		// whether the type is a floating point type, if it is widthExp may not be 0
	uint8_t vecExp:3;		// number of elements in the vector represented as 2^vecExp, Ex: 4 would be 2, 3 is special cased as 6
} StorageType;// __attribute__((packed));
*/
typedef struct {
	int16_t param[3];		// effects execution range and size of the output buffers, see rangeMode above
	uint16_t ref_idx;		// index of the arg whose size is the reference size that any relative size calculations will be based on
	enum rangeMode mode;	// what mode to calculate the NDRange/size_t[3] in
} RangeData;

// used to track fixed arg settings that stay constant between instances of a staged queue, regardless of image size
typedef struct {
	cl_mem_object_type type;// indicates what broad type of argument this should be
	RangeData size;			// data on how to calculate the size_t[3] of the arg
	cl_mem_flags flags;		// stores flag state to be assigned to eventual cl_mem object at creation, some from manifest, some from kernel arg queries
	cl_image_format format;	// used for verifying compatible channel types, spacing and read/write operations
} ArgStaging;	//TODO: since stbi only supports 8 bit depth the host readable flag forces 8 bit output which may cause calculation issues if buffer isn't last

// user provided info of how to set up kernels in a queue and their arguments
typedef struct {
	RangeData range;		// data on how to calculate the NDRange
	uint16_t kernel_idx;	// index of the kernel program name
	uint16_t* arg_idxs;		// array containing indices for each arg to use, freed when using freeStagingArray()	//TODO: write freeStagingArray()
} KernStaging;

typedef struct {
	uint16_t kernel_cnt;
	uint16_t stage_cnt;
	uint16_t img_arg_cnt;
	uint16_t input_img_cnt;
	char** kprog_names;		// array of kernel program names that are used and must be compiled
	KernStaging* kern_stg;	// kernel staging array listing all stages, their scheduling details, and their program indices
	char** arg_names;		// array of kernel program argument names that get used for the stages
	ArgStaging* img_arg_stg;// arg staging array listing details about type of arg, and size
} QStaging;
/*
// info used in assigning an arg to kernels, creating/reading buffers on the host, and deallocating mem objects
typedef struct {
	cl_mem arg;				// pointer to the image or generic cl_mem object in question
	size_t size[3];			// used for calculating later arg sizes and host buffer sizes
	cl_image_format format;	// used for verifying compatible channel types, spacing and read/write operations
} TrackedArg;

typedef struct {
	TrackedArg* args;
//	uint16_t args_cnt;
//	uint16_t max_args;
	size_t max_out_size;
} ArgTracker;
*/
// info actually used in enqueueing kernels
typedef struct {
	// counts duplicated from QStaging so that it may safely have its contents freed and go out of scope
	uint16_t stage_cnt;	// how many stages the kernel array contains
	uint16_t img_arg_cnt;	// how many args the img_args array contains
	cl_kernel* kernels;	// array of kernel instances corresponding to each stage
	Size3D* ranges;		// array of 3D ranges to enque the matching kernel index with
	cl_mem* img_args;	// array of all image args associated with the kernel
	Size3D* img_sizes;	// array of images sizes corresponding to each arg
//	char name[32];		// name of the kernel function, only used for user convenience/debugging
} StagedQ;

#endif//CLBP_PUBLIC_TYPEDEFS_H