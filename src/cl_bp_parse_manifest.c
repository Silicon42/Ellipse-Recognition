
#include "cl_bp_parse_manifest.h"

#define MANIFEST_ERROR "\nMANIFEST ERROR: "

//TODO: unify the error handling with a constant string array and meaningful return values
inline bool printMissingArgEntry(const char* arg_name)
{
	fprintf(stderr, MANIFEST_ERROR"Requested arg \"%s\" but no such key was found under [args].\n", arg_name);
	return true;
}

bool printInvalidStorageType(const char* arg_name)
{
	fprintf(stderr, MANIFEST_ERROR"[args] %s has invalid type.\n", arg_name);
	return true;
}

inline bool printInvalidArgType(const char* arg_name)
{
	fprintf(stderr, MANIFEST_ERROR"[args] %s has invalid argument type.\n", arg_name);
	return true;
}

bool parseRangeData(const char** arg_name_list, int arg_cnt, RangeData* ret, toml_table_t* size_tbl)
{
	if(!size_tbl)	// size wasn't specified, fallback to default
	{
		*ret = (RangeData){
			.param = {0,0,0},
			.ref_idx = arg_cnt - 1,
			.mode = REL
		};
		return false;
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
		{
			fprintf(stderr, MANIFEST_ERROR"referenced arg \"%s\" for size but it's not staged at this point.", val.u.s);
			return true;
		}
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

// returns true on validation failure
bool validateNstoreArgConfig(const char** arg_name_list, ArgStaging* arg_stg, int arg_stg_cnt, toml_table_t* args, char* arg_name)
{
	toml_table_t* arg_conf = toml_table_table(args, arg_name);
	if(!arg_conf)
		return printMissingArgEntry(arg_name);

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
		return printInvalidStorageType(arg_name);
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
		return printInvalidStorageType(arg_name);

	// check type's vector size if any
	switch(storage.u.s[pos])
	{
	default:
		return printInvalidStorageType(arg_name);
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
	default:
		return printInvalidArgType(arg_name);
	case 'b':	// buffer		//TODO: implement the rest of these types
	case 'p':	// pipe
	case 's':	// scalar
	case 'a':	// array (image)
	case 'i':	// image
		new_arg->type = type.u.s[0];
	}
	
	toml_table_t* size_tbl = toml_table_table(arg_conf, "size");
}
