#include <stdio.h>
#include <CL/cl.h>
#define CL_DEVICE_HALF_FP_CONFIG 0x1033
#include "common_error_handlers.h"

struct device_list{
	cl_uint device_cnt;
	cl_device_id* devices;
};

const char platInfo[] = "clGetPlatformInfo";
const char devInfo[] = "clGetDeviceInfo";

void mystrchr(char* str, char match, char replace)
{
	while(*str != '\0')
	{
		if(*str == match)
			*str = replace;
		++str;
	}
}

int main()
{
	// Host/device data structures //
	cl_uint platform_cnt = 0;
	cl_platform_id* platforms;
	struct device_list* devLists;
	cl_int clErr;

	// get the platform count for installed OpenCl SDKs
	clErr = clGetPlatformIDs(0, NULL, &platform_cnt);
	
	printf("Found %u OpenCL platforms:\n", platform_cnt);

	// allocate and fill platform list
	platforms = (cl_platform_id*) malloc(platform_cnt * sizeof(cl_platform_id));
	devLists = (struct device_list*) malloc(platform_cnt * sizeof(struct device_list));
	clErr = clGetPlatformIDs(platform_cnt, platforms, NULL);
	// should be sufficient to only check for errors here since the first call to clGetPlatformIDs should never error out
	handleClError(clErr, "clGetPlatformIDs");

	char buffChars[4096] = {0};
	cl_device_type devType;
	cl_uint buffUint;
	cl_ulong buffUlong;
	cl_bool buffBool;
	cl_device_fp_config buffFPConf;
	cl_device_mem_cache_type buffCacheType;
	cl_device_local_mem_type buffLocalType;
	size_t buffSizeTs[16] = {0};

	for(cl_uint i = 0; i < platform_cnt; ++i)
	{
		// print diagnostic platform info
		clErr = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 4096, buffChars, NULL);
		handleClError(clErr, platInfo);
		printf("\n============= Platform[%u] %s =============\n", i, buffChars);
		clErr = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 4096, buffChars, NULL);
		handleClError(clErr, platInfo);
		printf("Vendor: %s\n", buffChars);
		clErr = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 4096, buffChars, NULL);
		handleClError(clErr, platInfo);
		printf("OpenCL ver.: %s\n", buffChars);


		// enumerate the available devices
		clErr = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &devLists[i].device_cnt);

		devLists[i].devices = (cl_device_id*) malloc(devLists[i].device_cnt * sizeof(cl_device_id));
		clErr = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, devLists[i].device_cnt, devLists[i].devices, NULL);
		handleClError(clErr, "clGetDeviceIDs");

		for(cl_uint j = 0; j < devLists[i].device_cnt; ++j)
		{
			cl_device_id device = devLists[i].devices[j];
			clGetDeviceInfo(device, CL_DEVICE_NAME, 4096, buffChars, NULL);
			handleClError(clErr, devInfo);
			printf("\n++++ Device[%u]: %s ++++\n", j, buffChars);
			clGetDeviceInfo(device, CL_DEVICE_VERSION, 4096, buffChars, NULL);
			handleClError(clErr, devInfo);
			printf("Device OpenCL verision: %s\n", buffChars);
			clGetDeviceInfo(device, CL_DEVICE_PROFILE, 4096, buffChars, NULL);
			handleClError(clErr, devInfo);
			printf("Profile: %s\n", buffChars);
			clGetDeviceInfo(device, CL_DRIVER_VERSION, 4096, buffChars, NULL);
			handleClError(clErr, devInfo);
			printf("Driver: %s\n", buffChars);
			// device type info
			clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &devType, NULL);
			handleClError(clErr, devInfo);
			printf("CPU: %i	GPU: %i	ACC: %i	Default: %i\n", \
				!!(devType & CL_DEVICE_TYPE_CPU), \
				!!(devType & CL_DEVICE_TYPE_GPU), \
				!!(devType & CL_DEVICE_TYPE_ACCELERATOR), \
				!!(devType & CL_DEVICE_TYPE_DEFAULT));
			clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("Max clock freq. (MHz):	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(size_t), buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("Profiling timer resolution (ns): %zu\n", *buffSizeTs);

			// work size limitations
			printf("\n--- Compute Size ---\n");
			clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("Max compute units:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("Max work item dims:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*16, buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("Max work item dim sizes:	");
			for(cl_uint k = 0; k < buffUint; ++k)
				printf("%zu, ", buffSizeTs[k]);

			clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("\nMax work group size *: %zu\n", *buffSizeTs);
			printf(" * May not be preferred size for kernel,\n Refer to clGetKernelWorkGroupInfo() with query CL_KERNEL_WORK_GROUP_SIZE\n");
			
			// type/vector limitations
			printf("\n--- Type Support ---\n");
			clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool), &buffBool, NULL);
			handleClError(clErr, devInfo);
			printf("Is little endian: %u\n", buffBool);

			printf("\nPreferred Vector Widths\n");
			clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("char:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("short:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("int:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("long:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("half:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("float:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("double:	%u\n", buffUint);

			printf("\nNative Vector Widths\n");
			clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("char:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("short:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("int:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("long:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("half:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("float:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("double:	%u\n", buffUint);

			clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG, sizeof(cl_device_fp_config), &buffFPConf, NULL);
			handleClError(clErr, devInfo);
			printf("\nHALF CONFIG:\nHardware: %i	FMA: %i	Denorm: %i	Inf/NaN: %i\nRound Near: %i	Round Zero: %i	Round Inf: %i	Correct Div/Sqrt Round: %i\n", \
				!(buffFPConf & CL_FP_SOFT_FLOAT), \
				!!(buffFPConf & CL_FP_FMA), \
				!!(buffFPConf & CL_FP_DENORM), \
				!!(buffFPConf & CL_FP_INF_NAN), \
				!!(buffFPConf & CL_FP_ROUND_TO_NEAREST), \
				!!(buffFPConf & CL_FP_ROUND_TO_INF), \
				!!(buffFPConf & CL_FP_DENORM), \
				!!(buffFPConf & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT));

			clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &buffFPConf, NULL);
			handleClError(clErr, devInfo);
			printf("\nSINGLE CONFIG:\nHardware: %i	FMA: %i	Denorm: %i	Inf/NaN: %i\nRound Near: %i	Round Zero: %i	Round Inf: %i	Correct Div/Sqrt Round: %i\n", \
				!(buffFPConf & CL_FP_SOFT_FLOAT), \
				!!(buffFPConf & CL_FP_FMA), \
				!!(buffFPConf & CL_FP_DENORM), \
				!!(buffFPConf & CL_FP_INF_NAN), \
				!!(buffFPConf & CL_FP_ROUND_TO_NEAREST), \
				!!(buffFPConf & CL_FP_ROUND_TO_INF), \
				!!(buffFPConf & CL_FP_DENORM), \
				!!(buffFPConf & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT));

			clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &buffFPConf, NULL);
			handleClError(clErr, devInfo);
			printf("\nDOUBLE CONFIG:\nHardware: %i	FMA: %i	Denorm: %i	Inf/NaN: %i\nRound Near: %i	Round Zero: %i	Round Inf: %i	Correct Div/Sqrt Round: %i\n", \
				!(buffFPConf & CL_FP_SOFT_FLOAT), \
				!!(buffFPConf & CL_FP_FMA), \
				!!(buffFPConf & CL_FP_DENORM), \
				!!(buffFPConf & CL_FP_INF_NAN), \
				!!(buffFPConf & CL_FP_ROUND_TO_NEAREST), \
				!!(buffFPConf & CL_FP_ROUND_TO_INF), \
				!!(buffFPConf & CL_FP_DENORM), \
				!!(buffFPConf & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT));

			// mem + arg limitations
			printf("\n--- Mem + Arg limitations ---\n");
			clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("Address bits:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &buffUlong, NULL);
			handleClError(clErr, devInfo);
			printf("Max mem alloc:	%llu\n", buffUlong);
			clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &buffBool, NULL);
			handleClError(clErr, devInfo);
			printf("Image support:	%u\n", buffBool);
			clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("Max read image args:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("Max write image args:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("Max image2d width:	%zu\n", *buffSizeTs);
			clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("Max image2d height:	%zu\n", *buffSizeTs);
			clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("Max image3d width:	%zu\n", *buffSizeTs);
			clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("Max image3d height:	%zu\n", *buffSizeTs);
			clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("Max image3d depth:	%zu\n", *buffSizeTs);
			clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(size_t), buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("Max image1d pixels from buffer object:	%zu\n", *buffSizeTs);
			clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, sizeof(size_t), buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("Max images in 1d/2d image array:	%zu\n", *buffSizeTs);
			clGetDeviceInfo(device, CL_DEVICE_MAX_SAMPLERS, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("Max samplers:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(cl_uint), buffSizeTs, NULL);
			handleClError(clErr, devInfo);
			printf("Max kernel arg size:	%zu\n", *buffSizeTs);
			clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("Address alignment in bytes:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cl_device_mem_cache_type), &buffCacheType, NULL);
			handleClError(clErr, devInfo);
			printf("Cache type:	");
			if(buffCacheType == CL_READ_WRITE_CACHE)
				printf("Read write\n");
			else if(buffCacheType == CL_READ_ONLY_CACHE)
				printf("Read only\n");
			else
				printf("None\n");
			clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("Global mem cache line size:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &buffUlong, NULL);
			handleClError(clErr, devInfo);
			printf("Global mem cache size:	%llu\n", buffUlong);
			clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &buffUlong, NULL);
			handleClError(clErr, devInfo);
			printf("Global mem size:	%llu\n", buffUlong);
			clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &buffUlong, NULL);
			handleClError(clErr, devInfo);
			printf("Max constant buffer size:	%llu\n", buffUlong);
			clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(cl_uint), &buffUint, NULL);
			handleClError(clErr, devInfo);
			printf("Max constant args:	%u\n", buffUint);
			clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &buffLocalType, NULL);
			handleClError(clErr, devInfo);
			printf("Local mem type:	");
			if(buffLocalType == CL_LOCAL)
				printf("Local\n");
			else if(buffLocalType == CL_GLOBAL)
				printf("Global\n");
			else
				printf("None\n");
			clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &buffUlong, NULL);
			handleClError(clErr, devInfo);
			printf("Local mem size:	%llu\n", buffUlong);
			clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &buffBool, NULL);
			handleClError(clErr, devInfo);
			printf("Is host unified mem:	%u\n", buffBool);

			// kernels + extensions
			printf("\n--- Built-in Kernels ---\n");
			clGetDeviceInfo(device, CL_DEVICE_BUILT_IN_KERNELS, 4096, buffChars, NULL);
			handleClError(clErr, devInfo);
			mystrchr(buffChars, ';', '\n');
			printf("%s\n", buffChars);

			printf("\n--- Extensions ---\n");
			clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 4096, buffChars, NULL);
			handleClError(clErr, devInfo);
			mystrchr(buffChars, ' ', '\n');
			printf("%s\n", buffChars);

		}
	}
}
