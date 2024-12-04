#include "toml-c.h"
#include "cl_bp_public_typedefs.h"

// Validates a newly referenced entry in the args table and sets up a
// corresponding ArgStaging array entry
// returns true on validation failure
bool validateNstoreArgConfig(ArgStaging* arg_stg, int last_arg_idx, toml_table_t* args, char* arg_name);
