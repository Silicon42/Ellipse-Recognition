#include "toml-c.h"
#include "clbp_public_typedefs.h"

toml_table_t* parseManifestFile(char* fname, clbp_Error* e);
void allocQStagingArrays(const toml_table_t* root_tbl, QStaging* staging, clbp_Error* e);
void populateQStagingArrays(const toml_table_t* root_tbl, QStaging* staging, clbp_Error* e);