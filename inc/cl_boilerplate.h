#include <CL/cl.h>

// attempts to get the first available GPU or if none available CPU
//TODO: actually implement multiple attempts to find a GPU, currently just takes the first device of the first platform
cl_device_id getPreferredDevice();
cl_program buildProgramFromFile(cl_context context, cl_device_id device, const char* fname, const char* args);
// creates a single channel cl_mem image from a file
cl_mem imageFromFile(cl_context context, const char* fname, const cl_image_format* img_format, size_t* img_size);
cl_mem imageOutputBuffer(cl_context context, char** out_data, const cl_image_format* img_format, const size_t* img_size);
cl_mem imageIntermediateBuffer(cl_context context, const size_t* img_size, cl_channel_order order);
