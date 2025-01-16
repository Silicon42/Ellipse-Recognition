
#include "clbp_parse_manifest.h"
#include "cl_boilerplate.h"
#include "clbp_utils.h"
#include <assert.h>

clbp_Error parseRangeData(QStaging* staging, RangeData* ret, toml_table_t* size_tbl)
{
	// default values are used when they are not specified or are otherwise blank or the table itself is omitted
	*ret = (RangeData){
		.ref_idx = staging->img_arg_cnt - 1,
		.mode = CLBP_RM_ADD_SUB,
		.param = {0,0,0}
	};

	if(!size_tbl)	// size/range wasn't specified, fallback to default
		return (clbp_Error){0};

	//read the name of the reference arg
	toml_value_t val = toml_table_string(size_tbl, "ref_arg");
	if(val.u.s[0])	//string defaults to empty string if val.ok == false
	{
		int ref_idx = getStringIndex((char const**)staging->arg_names, val.u.s);
		// if the string wasn't in the list, it might have been referenced out of order
		// or mis-typed or completely missing, in any case we can't determine size from the given name
		if(ref_idx < 0)
			return (clbp_Error){.err_code = CLBP_MF_REF_ARG_NOT_YET_STAGED, .detail = val.u.s};	//FIXME: this could lead to segfault if freed while returning up the stack
		
		ret->ref_idx = ref_idx;
	}

	val = toml_table_string(size_tbl, "mode");
	if(val.u.s[0])
	{
		int mode = getStringIndex(modeNames, val.u.s);
		if(mode < 0)
			return (clbp_Error){.err_code = CLBP_MF_INVALID_RANGEMODE, .detail = val.u.s};	//FIXME: this could lead to segfault if freed while returning up the stack

		ret->mode = mode;
	}

	toml_array_t* params = toml_table_array(size_tbl, "params");
	if(params)
	{
		int dims = (params->nitem <= 3) ? params->nitem : 3;
		// assumes params start out as zero by default
		for(int i = 0; i < dims; ++i)
		{
			ret->param[i] = toml_array_int(params, i).u.i;
		}
	}

	return (clbp_Error){0};
}

clbp_Error validateNstoreArgConfig(QStaging* staging, toml_table_t* args, char* arg_name)
{
	toml_table_t* arg_conf = toml_table_table(args, arg_name);
	if(!arg_conf)
		return (clbp_Error){.err_code = CLBP_MF_MISSING_ARG_ENTRY, .detail = arg_name};

	uint16_t* arg_cnt = &staging->img_arg_cnt;
	ArgStaging* new_arg = &staging->img_arg_stg[*arg_cnt];

	// parse if arg was manually set to host readable, defaults to false if not specified
	toml_value_t is_host_readable = toml_table_bool(arg_conf, "is_host_readable");
	new_arg->flags = is_host_readable.u.b ? CL_MEM_HOST_READ_ONLY : 0;	// toml not ok should default to false for bool I think

	toml_value_t ch_type_toml = toml_table_string(arg_conf, "channel_type");
	enum clChannelType ch_type = CLBP_INVALID_CHANNEL_TYPE;
	if(ch_type_toml.u.s[0] != '\0')
		ch_type = getStringIndex(channelTypes, ch_type_toml.u.s) + CLBP_OFFSET_CHANNEL_TYPE;

	if(ch_type >= CLBP_INVALID_CHANNEL_TYPE || ch_type < CLBP_OFFSET_CHANNEL_TYPE)
		return (clbp_Error){.err_code = CLBP_MF_INVALID_CHANNEL_TYPE, .detail = arg_name};

	
	// infer order from channel count
	toml_value_t ch_cnt = toml_table_int(arg_conf, "channel_count");
	// clamp channel count to 1 thru 4 if not packed, 3 if packed and not 101010_2, and 4 if 101010_2
	uint8_t min_channels = 1;
	if(isChannelTypePacked(ch_type))
		min_channels = 3 + (ch_type == CLBP_UNORM_INT_101010_2);
	
	if(ch_cnt.u.i < min_channels)
		ch_cnt.u.i = min_channels;
	else if(ch_cnt.u.i > 4)
		ch_cnt.u.i = 4;
	
	enum clChannelOrder ch_order = getOrderFromChannelCnt(ch_cnt.u.i);

	new_arg->format = (cl_image_format){.image_channel_data_type = ch_type, .image_channel_order = ch_order};

	toml_value_t mem_type_toml = toml_table_string(arg_conf, "type");
	enum clMemType mem_type = getStringIndex(memTypes, mem_type_toml.u.s) + CLBP_OFFSET_MEMTYPE;
	if(mem_type >= CLBP_INVALID_MEM_TYPE || mem_type < CLBP_OFFSET_MEMTYPE)
		return (clbp_Error){.err_code = CLBP_MF_INVALID_ARG_TYPE, .detail = arg_name};
	
	new_arg->type = mem_type;
	
	toml_table_t* size_tbl = toml_table_table(arg_conf, "size");
	clbp_Error ret = parseRangeData(staging, &staging->arg_size_calcs[*arg_cnt], size_tbl);
	++(*arg_cnt);
	return ret;
}

toml_table_t* parseManifestFile(char* fname, clbp_Error* e)
{
	assert(fname && e);
	// Read in the manifest for what kernels should be used
	char* manifest;
	manifest = readFileToCstring(fname, e);
	if(e->err_code)
		return NULL;
	
	static char errbuf[256];	//TODO: ugly but probably fine, might be an issue with multiple instances but if they all error at the same time you have bigger issues
	toml_table_t* root_tbl = toml_parse(manifest, errbuf, sizeof(errbuf));
	free(manifest);	// parsing works for whole document, so c-string is no longer needed
	if(!root_tbl)
		*e = (clbp_Error){.err_code = CLBP_MF_PARSING_FAILED, errbuf};

	return root_tbl;
}

// expects staging->input_img_cnt to be set already to the expected value
void allocQStagingArrays(const toml_table_t* root_tbl, QStaging* staging, clbp_Error* e)
{
	assert(root_tbl && staging && e);
	// get stages array and check valid size and type
	toml_array_t* stage_list = toml_table_array(root_tbl, "Stages");
	if(!stage_list || stage_list->kind != 't')
	{
		*e = (clbp_Error){.err_code = CLBP_MF_INVALID_STAGES_ARRAY};
		return;
	}
	uint16_t* stage_cnt = &staging->stage_cnt;
	*stage_cnt = stage_list->nitem;

	// get arg table
	toml_table_t* args_table = toml_table_table(root_tbl, "Args");
	if(!args_table || !args_table->ntab)
	{
		e->err_code = CLBP_MF_INVALID_ARGS_TABLE;
		return;
	}
	uint32_t max_defined_args = args_table->ntab + staging->input_img_cnt;

	// get hardcoded args array (typically inputs defined in the source code)
	toml_array_t* hardcoded_args_arr = toml_table_array(root_tbl, "HCInputArgs");
	// if we are expecting input images,
	if(staging->input_img_cnt)
	{	// the hardcoded args array must exist as a string array with the same number of entries as expected inputs
		if(!hardcoded_args_arr || hardcoded_args_arr->type != 's' || hardcoded_args_arr->nitem != staging->input_img_cnt)
		{
			*e = (clbp_Error){.err_code = CLBP_MF_INVALID_HC_ARGS_ARRAY, .detail = NULL + staging->input_img_cnt};
			return;
		}
	}

	// calculate name count so we only have a singular allocation for the kernel and arg name strings (both are char pointer arrays),
	// also used for range_calcs and arg_size_calcs (both are Size3D arrays)
	uint32_t names_cnt = *stage_cnt + max_defined_args;
	staging->kprog_names = calloc(names_cnt + 2, sizeof(char*));	//NOTE: +2 for NULL pointer termination of 2 arrays
	staging->arg_names = staging->kprog_names + *stage_cnt + 1;
	staging->range_calcs = malloc(names_cnt * sizeof(Size3D));
	staging->arg_size_calcs = &staging->range_calcs[*stage_cnt];
	staging->input_imgs = calloc(staging->input_img_cnt, sizeof(char*));
	//TODO: check if I assumed the following elements were zeroed, if not, they can be malloc'd instead
	staging->kern_stg = calloc(*stage_cnt, sizeof(KernStaging));
	staging->img_arg_stg = calloc(max_defined_args, sizeof(ArgStaging));

	// check if any of the allocations failed and if so release any allocated componenets
	if(!staging->kprog_names || !staging->kern_stg || !staging->img_arg_stg || !staging->range_calcs || (staging->input_img_cnt && !staging->input_imgs))
	{
		free(staging->input_imgs);
		free(staging->kern_stg);
		free(staging->img_arg_stg);
		free(staging->range_calcs);
		free(staging->kprog_names);
		*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "Staging array allocation"};
		return;
	}

	// set the names of the hard-coded args while we have a reference to the toml_array_t
	for(int i = 0; i < staging->input_img_cnt; ++i)
	{
		toml_value_t name = toml_array_string(hardcoded_args_arr, i);
		if(name.u.s[0] == '\0')
		{
			*e = (clbp_Error){.err_code = CLBP_MF_INVALID_ARG_NAME, .detail = NULL + i};
			return;
		}
		staging->arg_names[i] = name.u.s;
		// set the flag for the hardcoded arg to copy the contents of host memory to device memory
		staging->img_arg_stg[i].flags = CL_MEM_COPY_HOST_PTR;
	}
	//TODO: check if it max_defined_args is still needed to be copied here or if this can be set to the hardcoded args count
	//technically this is an upper limit but it can be stored here temporarily until we get the real count
	//ADDENDUM: it's not but it would require a couple other changes to remove it so it stays for now
	staging->img_arg_cnt = max_defined_args;
	return;
}

// validate MANIFEST.toml and populate program list, kernel queue staging array, and arg staging
void populateQStagingArrays(const toml_table_t* root_tbl, QStaging* staging, clbp_Error* e)
{
	assert(root_tbl && staging && e);
	int max_defined_args = staging->img_arg_cnt;
	staging->kernel_cnt = 0;
	staging->img_arg_cnt = staging->input_img_cnt;

	// assumes stage list and args table was already validated
	toml_array_t* stage_list = toml_table_array(root_tbl, "Stages");
	toml_table_t* args_table = toml_table_table(root_tbl, "Args");

	KernStaging* curr_stage;
	// for each stage in the stage list
	for(int i = 0; i < staging->stage_cnt; ++i)
	{
		toml_table_t* stage = toml_array_table(stage_list, i);	//can't return null since we already have valid stage count
		toml_value_t tval = toml_table_string(stage, "name");
		if(!tval.u.s[0])	// with the change to toml-c.h, should be safe just to check for empty string
		{
			*e = (clbp_Error){.err_code = CLBP_MF_MISSING_STAGE_NAME, .detail = NULL + i};
			return;
		}

		// check if a kernel by that name already exists, if not, add it to the list of ones to build
		// additionally set the kernel program reference index for the stage to the returned index of the match/new program name
		curr_stage = &staging->kern_stg[i];
		int kern_idx = addUniqueString(staging->kprog_names, staging->stage_cnt, tval.u.s);
		curr_stage->kernel_idx = kern_idx;
		if(staging->kernel_cnt == kern_idx)	//check if this was a newly referenced kernel
			++staging->kernel_cnt;
		//FIXME: ^ something must eventually copy the string or you'll have a read after free for the toml strings
		// alternatively, figure out how to parse the toml values in place such that their contents fit into the space of the
		// original file

		toml_array_t* stage_args = toml_table_array(stage, "args");
		if(!stage_args || stage_args->kind != 'v' || stage_args->type != 's')
		{
			*e = (clbp_Error){.err_code = CLBP_MF_INVALID_STAGE_ARGS_ARRAY, .detail = NULL + i};
			return;
		}
		int args_cnt = stage_args->nitem;

		curr_stage->arg_idxs = malloc(args_cnt * sizeof(uint16_t));
		if(!curr_stage->arg_idxs)
		{
			*e = (clbp_Error){.err_code = CLBP_OUT_OF_MEMORY, .detail = "stage's argument index array"};
			return;
		}

		int stg_img_arg_cnt = toml_array_len(stage_args);
		uint16_t* curr_arg_idx;
		// iterate over args to find any new ones
		for(int j = 0; j < stg_img_arg_cnt; ++j)
		{
			curr_arg_idx = &curr_stage->arg_idxs[j];
			char* arg_name = toml_array_string(stage_args, j).u.s;	//guaranteed exists due to kind, and type checks above
			if(arg_name[0])	// if not empty string
			{
				int arg_idx = addUniqueString(staging->arg_names, max_defined_args, arg_name);
				*curr_arg_idx = arg_idx;
				if(staging->img_arg_cnt == arg_idx)	//check if this was a newly referenced argument
				{
					//staging->img_arg_cnt = arg_idx + 1;
					//instantiate a corresponding arg on the arg staging array

					*e = validateNstoreArgConfig(staging, args_table, arg_name);
					if(e->err_code != CLBP_OK)
						return;
				}
			}
			else	// empty string is a special case that always selects whatever was last added
				*curr_arg_idx = staging->img_arg_cnt - 1;
		}

		toml_table_t* range = toml_table_table(stage, "range");
		*e = parseRangeData(staging, &staging->range_calcs[i], range);
		if(e->err_code)
			return;
	}
}
