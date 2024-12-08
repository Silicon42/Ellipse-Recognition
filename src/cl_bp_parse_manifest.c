
#include "cl_bp_parse_manifest.h"
#include "cl_boilerplate.h"

cl_bp_Error parseRangeData(const char** arg_name_list, int arg_cnt, RangeData* ret, toml_table_t* size_tbl)
{
	if(!size_tbl)	// size wasn't specified, fallback to default
	{
		*ret = (RangeData){
			.param = {0,0,0},
			.ref_idx = arg_cnt - 1,
			.mode = REL
		};
		return (cl_bp_Error){0};
	}

	//read the name of the reference arg
	toml_value_t val = toml_table_string(size_tbl, "ref_arg");
	if(!val.u.s[0])	//string defaults to empty string if val.ok == false
		ret->ref_idx = arg_cnt - 1;	// if missing or empty, default to previous arg
	else
	{
		ret->ref_idx = getStringIndex(arg_name_list, val.u.s);
		// if the string wasn't in the list, it might have been referenced out of order
		// or mis-typed or completely missing, in any case we can't determine size from the given name
		if(ret->ref_idx < 0)
			return (cl_bp_Error){.err_code = CL_BP_MF_REF_ARG_NOT_YET_STAGED, .detail = val.u.s};
	}

	toml_array_t* params = toml_table_array(size_tbl, params);
	// assumes params start out as zero by default
	if(params)
	{
		ret->param[0] = 0;
		ret->param[1] = 0;
		ret->param[2] = 0;
	}

	val = toml_table_string(size_tbl, "mode");
}

cl_bp_Error validateNstoreArgConfig(const char** arg_name_list, ArgStaging* arg_stg, int arg_stg_cnt, toml_table_t* args, char* arg_name)
{
	toml_table_t* arg_conf = toml_table_table(args, arg_name);
	if(!arg_conf)
		return (cl_bp_Error){.err_code = CL_BP_MF_MISSING_ARG_ENTRY, .detail = arg_name};

	ArgStaging* new_arg = &arg_stg[arg_stg_cnt];
	toml_value_t storage = toml_table_string(arg_conf, "storage");
	StorageType st = {0};
	int pos = 0;
	// set the storage type flags
	switch(storage.u.s[0])
	{
	case 'u':	//	uchar, ushort, uint, ulong
		st.isUnsigned = true;
		++pos;
		break;
	case 'q':	// quad
	case 'd':	// double
	case 'f':	// float
	case 'h':	// half
		st.isFloat = true;
	}

	// set the width exponent of the type and advance pos to the character where the vector size would be specified
	switch(storage.u.s[pos])
	{
	default:	// invalid
		return (cl_bp_Error){.err_code = CL_BP_MF_INVALID_STORAGE_TYPE, .detail = arg_name};
	case 'c':	// char
		st.widthExp = 0;
		pos += 4;
		break;
	case 's':	// short
		st.widthExp = 1;
		pos += 5;
		break;
	case 'i':	// int
		st.widthExp = 2;
		pos += 3;
		break;
	case 'l':	// long
		st.widthExp = 3;
		pos += 4;
		break;
	case 'h':	// half
		st.widthExp = 1;
		pos += 4;
		break;
	case 'f':	// float
		st.widthExp = 2;
		pos += 5;
		break;
	case 'd':	// double
		st.widthExp = 3;
		pos += 6;
		break;
	case 'q':	// quad
		st.widthExp = 4;
		pos += 4;
	}

	// check that we won't potentially access past the end of the string
	if(storage.sl < pos)
		return (cl_bp_Error){.err_code = CL_BP_MF_INVALID_STORAGE_TYPE, .detail = arg_name};

	// check type's vector size if any
	switch(storage.u.s[pos])
	{
	default:	//invalid
		return (cl_bp_Error){.err_code = CL_BP_MF_INVALID_STORAGE_TYPE, .detail = arg_name};
	case '2':	// 2
		st.vecExp = 1;
		break;
	case '4':	// 4
		st.vecExp = 2;
		break;
	case '8':	// 8
		st.vecExp = 3;
		break;
	case '1':	// 16
		st.vecExp = 4;
		break;
	case '3':	// 3 (special case)
		st.vecExp = 6;
	case '\0':	// single item/not a vector
	}
	new_arg->storage_type = st;

	toml_value_t type = toml_table_string(arg_conf, "type");
	switch(type.u.s[0])
	{
	default:	// invalid
		return (cl_bp_Error){.err_code = CL_BP_MF_INVALID_ARG_TYPE, .detail = arg_name};
	case 'b':	// buffer		//TODO: implement the rest of these types
	case 'p':	// pipe
	case 's':	// scalar
	case 'a':	// array (image)
	case 'i':	// image
		new_arg->type = type.u.s[0];
	}
	
	toml_table_t* size_tbl = toml_table_table(arg_conf, "size");

	return (cl_bp_Error){0};
}

cl_bp_Error parseManifestFile(const char* fname)
{
	// Read in the manifest for what kernels should be used
	char* manifest;
	cl_bp_Error ret = readFileToCstring(fname, manifest);
	if(ret.err_code)
		return ret;
	
	char errbuf[256];
	toml_table_t* root_tbl = toml_parse(manifest, errbuf, sizeof(errbuf));
	free(manifest);	// parsing works for whole document, so c-string is no longer needed
	if(!root_tbl)
		return ret = (cl_bp_Error){.err_code = CL_BP_MF_PARSING_FAILED, errbuf};	//FIXME: returning errbuf pointer unsafe as is, READ AFTER SCOPE EXIT

	// get stages array and check valid size and type
	toml_array_t* stage_list = toml_table_array(root_tbl, "stages");
	if(!stage_list || stage_list->kind != 't')
		return ret = (cl_bp_Error){.err_code = CL_BP_MF_INVALID_STAGES_ARRAY};

	int stage_cnt = stage_list->nitem;

	// get arg table
	toml_table_t* args_table = toml_table_table(root_tbl, "args");
	if(!args_table || !args_table->nkval)
		return ret = (cl_bp_Error){.err_code = CL_BP_MF_INVALID_ARGS_TABLE};
	int max_defined_args = args_table->nkval;


}