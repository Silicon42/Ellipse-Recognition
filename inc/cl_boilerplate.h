#ifndef CL_BOILERPLATE_H
#define CL_BOILERPLATE_H

//#define CL_VERSION_2_0
//#define CL_VERSION_2_1
#include <CL/cl.h>

enum rangeMode {
//	PAD,	// pad to multiple of work group dimensions
	EXACT,	// set range/size to param
	SINGLE,	// meant for primarily serial workloads, [0] == false -> 1 hardware workgroup, [0] == true -> single work item
	REL,	// expand/contract range and output relative to input
	DIAG,	// exact on [0], contraction(- only, + no useful effect) relative to length of diagonal on [1]*, relative on [2], used for hough_lines
	DIVIDE,	// divides each component by corresponding param
};

typedef struct {
	enum rangeMode mode;	// what mode to calculate the NDRange/size_t[3] in
	int param[3];			// effects execution range and size of the output buffers, see rangeMode above
} RangeData;

typedef struct {
	uint16_t rel_ref;		// how many elements back relative to the current arg the operation is referencing
	RangeData range;		// data on how to calculate the size_t[3] of the arg
	char is_host_readable;	// boolean flag indicating if the host will read an output buffer, N/A for read args
	char is_array;			// boolean indicating if arg should treat last dimension as array, N/A for read args
} ArgStaging;	//TODO: since stbi only supports 8 bit depth the host readable flag forces 8 bit output which may cause calculation issues if buffer isn't last

// user provided info of how to set up kernels in a queue and their arguments
typedef struct {
	int kernel_idx;		// index of the reference kernel provided for cloning
	uint16_t rel_ref;	// how many elements back relative to the current end of the ArgTracker list the operation is referencing
	RangeData range;	// data on how to calculate the NDRange
	ArgStaging* args;	// array containing data on how to construct each arg
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

// attempts to get the first available GPU or if none available CPU
//TODO: actually implement multiple attempts to find a GPU, currently just takes the first device of the first platform
cl_device_id getPreferredDevice();

// reads in a list of files by the names of the kernel functions, builds them and makes a kernel for each in an array up to 
// max_kernels, returns the number of kernels actually created
cl_uint buildKernelsFromSource(cl_context context, cl_device_id device, const char* src_dir, const char** names, const char* args, cl_kernel* kernels, cl_uint max_kernels);
// takes a NULL terminated array of QStaging pointers and an array of kernels and fills in the QStage array and argTracker array
// according to their details
int prepQStages(cl_context context, const QStaging** staging, const cl_kernel* ref_kernels, QStage* stages, int max_stages, ArgTracker* at);
// creates a single channel cl_mem image from a file and attaches it to the tracked arg pointer provided
// the tracked arg must have the format pre-populated with a suitable way to interpret the raw image data
void imageFromFile(cl_context context, const char* fname, TrackedArg* tracked);


//----VVVV----UTILITY FUNCTIONS, NOT NECCESSARILY USEFUL TO USER----VVVV----//

cl_mem createImageBuffer(cl_context context, char is_host_readable, char is_array, const cl_image_format* img_format, const size_t img_size[3]);
// validates metadata[0 thru 2] formating and returns true if valid
char isArgMetadataValid(char* metadata);
cl_channel_type getTypeFromMetadata(const char* metadata, char isHostReadable);
cl_channel_order getOrderFromMetadata(const char* metadata);

void calcSizeByMode(const size_t* in, const RangeData* range, size_t* out);

char getDeviceRWType(cl_channel_type type);
char getArgStorageType(cl_channel_type type);
unsigned char getChannelCount(cl_channel_order order);
// this only issues warnings to the user since they could easily have misnamed it
// and it isn't required data unlike on writes that need new textures
void verifyReadArgTypeMatch(cl_image_format ref_format, char* metadata);

// converts an end relative arg index to a pointer to the referenced TrackedArg with error checking
TrackedArg* getRefArg(const ArgTracker* at, uint16_t rel_ref);

void setKernelArgs(cl_context context, const QStaging* stage, cl_kernel kernel, ArgTracker* at);


#endif//CL_BOILERPLATE_H