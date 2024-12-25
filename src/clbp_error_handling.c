#include <stdio.h>
#include "clbp_error_handling.h"
#include "cl_error_handlers.h"
#define MANIFEST_ERROR "\nMANIFEST ERROR: "

static const char* clbp_error_strings[] = {
	"\nNo Error.\n",
	"\nERROR: Out of memory, failed to allocate %s.\n",
	"\nERROR: Couldn't find file \"%s\".\n",
	"\nERROR: Invalid RangeMode at index %i (+: arg index, -: kernel index)\n",
	"\nERROR: Invalid RangeMode at index %i (+: arg index, -: kernel index)\n",

	"\nTOML ERROR: %s\n",
	MANIFEST_ERROR"Stages array must be a table array with at least one item.\n",
	MANIFEST_ERROR"Args table was empty.\n",
	MANIFEST_ERROR"Missing \"name\" key at entry %i of stages array.\n",
	MANIFEST_ERROR"Args array at entry %i of stages array must be an array of strings with at least one item.\n",
	MANIFEST_ERROR"Requested arg \"%s\" but no such key was found under [args].\n",
	MANIFEST_ERROR"[args] \"%s\" has invalid storage type.\n",
	MANIFEST_ERROR"[args] \"%s\" has invalid argument type.\n",
	MANIFEST_ERROR"Referenced arg \"%s\" for size but it is not staged prior to this point.\n"
};

// if err_code not CLBP_OK, prints the error message with details injected and
// exits with the error code number as program return value
void handleClBoilerplateError(clbp_Error e)
{
	if(e.err_code == CLBP_OK)
		return;
	else if(e.err_code > CLBP_OK)
		fprintf(stderr, clbp_error_strings[e.err_code], e.detail);
	else
		handleClError(e.err_code, e.detail);
	exit(e.err_code);
}