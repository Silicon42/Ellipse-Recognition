#include "toml-c.h"
#include "cl_bp_public_typedefs.h"

// Validates a newly referenced entry in the args table and sets up a
// corresponding ArgStaging array entry
// returns (cl_bp_Error){0} on success
cl_bp_Error validateNstoreArgConfig(const char** arg_name_list, ArgStaging* arg_stg, int arg_stg_cnt, toml_table_t* args, char* arg_name);
