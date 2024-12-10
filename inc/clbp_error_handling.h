
// all error values are 'true'
enum clbp_ErrCode{
	CLBP_OK = 0,			// No error
	CLBP_OUT_OF_MEMORY,	// failed to allocate memory
	CLBP_FILE_NOT_FOUND,	// failed when attempting to open file, could be it doesn't exist or permissions
	// manifest parsing specific errors, all should be >= CLBP_MF_PARSING_FAILED
	CLBP_MF_PARSING_FAILED,		// all toml-c errors get converted to this
	CLBP_MF_INVALID_STAGES_ARRAY,		// stages array is missing or invalid, ie not a table array or empty
	CLBP_MF_INVALID_ARGS_TABLE,		// args table is missing or invalid, ie empty
	CLBP_MF_MISSING_STAGE_NAME,		// stages must specify names since that identifies which file kernel program to use
	CLBP_MF_INVALID_STAGE_ARGS_ARRAY,	// stage is missing its args array or has non-string entries in the array
	CLBP_MF_MISSING_ARG_ENTRY,			// key by name of the requested arg is missing in the args table
	CLBP_MF_INVALID_STORAGE_TYPE,		// storage type specifier string didn't match a recognized type
	CLBP_MF_INVALID_ARG_TYPE,			// arg type specifier string didn't match a recognized type
	CLBP_MF_REF_ARG_NOT_YET_STAGED,	// a staged arg referenced an arg that was not staged before it, either it doesn't exist or

};

typedef struct {
	enum clbp_ErrCode err_code;
	char* detail;
} clbp_Error;

// if there is an error code, prints an error message and then exits with that error code;
void handleClBoilerplateError(clbp_Error e);
