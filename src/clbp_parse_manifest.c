
#include "clbp_parse_manifest.h"
#include "cl_boilerplate.h"

clbp_Error parseRangeData(char** arg_name_list, int arg_cnt, RangeData* ret, toml_table_t* size_tbl)
{
	if(!size_tbl)	// size wasn't specified, fallback to default
	{
		*ret = (RangeData){
			.param = {0,0,0},
			.ref_idx = arg_cnt - 1,
			.mode = REL
		};
		return (clbp_Error){0};
	}

	//read the name of the reference arg
	toml_value_t val = toml_table_string(size_tbl, "ref_arg");
	if(!val.u.s[0])	//string defaults to empty string if val.ok == false
		ret->ref_idx = arg_cnt - 1;	// if missing or empty, default to previous arg
	else
	{
		int ref_idx = getStringIndex(arg_name_list, val.u.s);
		// if the string wasn't in the list, it might have been referenced out of order
		// or mis-typed or completely missing, in any case we can't determine size from the given name
		if(ref_idx < 0)
			return (clbp_Error){.err_code = CLBP_MF_REF_ARG_NOT_YET_STAGED, .detail = val.u.s};
		
		ret->ref_idx = ref_idx;
	}

	toml_array_t* params = toml_table_array(size_tbl, "params");
	// assumes params start out as zero by default
	if(params)
	{
		ret->param[0] = 0;
		ret->param[1] = 0;
		ret->param[2] = 0;
	}

	val = toml_table_string(size_tbl, "mode");
	return (clbp_Error){0};
}

clbp_Error validateNstoreArgConfig(char** arg_name_list, ArgStaging* arg_stg, int arg_cnt, toml_table_t* args, char* arg_name)
{
	toml_table_t* arg_conf = toml_table_table(args, arg_name);
	if(!arg_conf)
		return (clbp_Error){.err_code = CLBP_MF_MISSING_ARG_ENTRY, .detail = arg_name};

	ArgStaging* new_arg = &arg_stg[arg_cnt];
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
		return (clbp_Error){.err_code = CLBP_MF_INVALID_STORAGE_TYPE, .detail = arg_name};
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
		return (clbp_Error){.err_code = CLBP_MF_INVALID_STORAGE_TYPE, .detail = arg_name};

	// check type's vector size if any
	switch(storage.u.s[pos])
	{
	default:	//invalid
		return (clbp_Error){.err_code = CLBP_MF_INVALID_STORAGE_TYPE, .detail = arg_name};
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
		return (clbp_Error){.err_code = CLBP_MF_INVALID_ARG_TYPE, .detail = arg_name};
	case 'b':	// buffer		//TODO: implement the rest of these types
	case 'p':	// pipe
	case 's':	// scalar
	case 'a':	// array (image)
	case 'i':	// image
		new_arg->type = type.u.s[0];
	}
	
	toml_table_t* size_tbl = toml_table_table(arg_conf, "size");
	return parseRangeData(arg_name_list, arg_cnt, &new_arg->size, size_tbl);

}

toml_table_t* parseManifestFile(char* fname, clbp_Error* e)
{
	assert(fname && e);
	// Read in the manifest for what kernels should be used
	char* manifest;
	clbp_Error ret;
	manifest = readFileToCstring(fname, &ret);
	if(ret.err_code)
		return NULL;
	
	static char errbuf[256];	//TODO: ugly but probably fine, might be an issue with multiple instances but if they all error at the same time you have bigger issues
	toml_table_t* root_tbl = toml_parse(manifest, errbuf, sizeof(errbuf));
	free(manifest);	// parsing works for whole document, so c-string is no longer needed
	if(!root_tbl)
		*e = (clbp_Error){.err_code = CLBP_MF_PARSING_FAILED, errbuf};

	return root_tbl;
}

void allocQStagingArrays(const toml_table_t* root_tbl, QStaging* staging, clbp_Error* e)
{
	assert(root_tbl && staging && e);
	// get stages array and check valid size and type
	toml_array_t* stage_list = toml_table_array(root_tbl, "stages");
	if(!stage_list || stage_list->kind != 't')
	{
		*e = (clbp_Error){.err_code = CLBP_MF_INVALID_STAGES_ARRAY};
		return;
	}
	int stage_cnt = stage_list->nitem;
	staging->kprog_names = calloc(stage_cnt + 1, sizeof(char*));
	if(!staging->kprog_names)
	{
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "kernel program names array"};
		return;
	}
	staging->kern_stg = calloc(stage_cnt, sizeof(KernStaging));
	if(!staging->kern_stg)
	{
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "KernStaging array"};
		return;
	}
	staging->stage_cnt = stage_cnt;

	// get arg table
	toml_table_t* args_table = toml_table_table(root_tbl, "args");
	if(!args_table || !args_table->nkval)
	{
		*e = (clbp_Error){.err_code = CLBP_MF_INVALID_ARGS_TABLE};
		return;
	}
	int max_defined_args = args_table->nkval;

	staging->arg_names = calloc(max_defined_args + 1, sizeof(char*));
	if(!staging->arg_names)
	{
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "kernel arguments names array"};
		return;
	}
	staging->arg_stg = calloc(max_defined_args, sizeof(ArgStaging));
	if(!staging->arg_stg)
	{
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "ArgStaging array"};
		return;
	}
	//technically this is an upper limit but it can be stored here temporarily until we get the real count
	staging->arg_cnt = max_defined_args;
	return;
}

// validate MANIFEST.toml and populate program list, kernel queue staging array, and arg staging
void populateQStagingArrays(const toml_table_t* root_tbl, QStaging* staging, clbp_Error* e)
{
	assert(root_tbl && staging && e);
	int max_defined_args = staging->arg_cnt;
	int arg_cnt = 0;

	// assumes stage list and args table was already validated
	toml_array_t* stage_list = toml_table_array(root_tbl, "stages");
	toml_table_t* args_table = toml_table_table(root_tbl, "args");

	for(int i = 0; i < staging->stage_cnt; ++i)
	{
		toml_table_t* stage = toml_array_table(stage_list, i);	//can't return null since we already have valid stage count
		toml_value_t tval = toml_table_string(stage, "name");
		if(!tval.ok || !tval.u.s[0])	// with the change to toml-c.h, should be safe just to check for empty string
		{
			*e = (clbp_Error){.err_code = CLBP_MF_MISSING_STAGE_NAME, .detail = NULL + i};
			return;
		}

		// check if a kernel by that name already exists, if not, add it to the list of ones to build
		// additionally set the kernel program reference index for the stage to the returned index of the match/new program name
		staging->kern_stg[i].kernel_idx = addUniqueString(staging->kprog_names, staging->stage_cnt, tval.u.s);
		//FIXME: ^ something must eventually copy the string or you'll have a read after free for the toml strings

		toml_array_t* stage_args = toml_table_array(stage, "args");
		if(!stage_args || stage_args->kind != 'v' || stage_args->type != 's')
		{
			*e = (clbp_Error){.err_code = CLBP_MF_INVALID_STAGE_ARGS_ARRAY, .detail = NULL + i};
			return;
		}
		int args_cnt = stage_args->nitem;

		staging->kern_stg[i].arg_idxs = malloc(args_cnt * sizeof(uint16_t));
		if(!staging->kern_stg[i].arg_idxs)
		{
			*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "stage's argument index array"};
			return;
		}

		int stg_arg_cnt = toml_array_len(stage_args);

		// iterate over args to find any new ones
		for(int j = 0; j < stg_arg_cnt; ++j)
		{
			char* arg_name = toml_array_string(stage_args, j).u.s;	//guaranteed exists due to kind, and type checks above
			if(arg_name[0])	// if not empty string
			{
				int arg_idx = addUniqueString(staging->arg_names, max_defined_args, arg_name);
				staging->kern_stg[i].arg_idxs[j] = arg_idx;
				if(arg_idx == arg_cnt)	//check if this was a newly referenced argument
				{
					++arg_cnt;	//is only ever bigger by one so this is safe
					//instantiate a corresponding arg on the arg staging array

					*e = validateNstoreArgConfig(staging->arg_names, staging->arg_stg, arg_cnt, args_table, arg_name);
					if(e->err_code != CLBP_OK)
						return;
				}
			}
			else	// empty string is a special case that always selects whatever was last added
				staging->kern_stg[i].arg_idxs[j] = arg_cnt - 1;
		}
	}

}
