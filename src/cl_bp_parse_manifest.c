
#include "cl_bp_parse_manifest.h"

#define MANIFEST_ERROR "\nMANIFEST ERROR: "

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

// returns true on validation failure
bool validateNstoreArgConfig(ArgStaging* arg_stg, int last_arg_idx, toml_table_t* args, char* arg_name)
{
	toml_table_t* arg_conf = toml_table_table(args, arg_name);
	if(!arg_conf)
		return printMissingArgEntry(arg_name);

	ArgStaging* new_arg = &arg_stg[last_arg_idx];
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
		new_arg->size.param;
	}
	//arg_stg[last_arg_idx];

}
