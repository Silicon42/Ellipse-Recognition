#ifndef LINK_MACROS_CL
#define LINK_MACROS_CL

#define HAS_R_CONT		(1 << 3)
#define HAS_L_CONT		(1 << 4)
#define HAS_BOTH_CONT	(HAS_L_CONT | HAS_R_CONT)
#define R_CONT_IDX_MASK	(HAS_R_CONT - 1)
#define END_ADJ_SHIFT	6
#define IS_END_ADJ		(1 << END_ADJ_SHIFT)
#define IS_START		(1 << 7)
#define L_CONT_IDX_SHIFT	5

//NOTE: returned values in link_edge_pexels.cl are in the form 0blllLRrrr where
// "L" is a flag indicating valid left connection data,
// "R" is a flag indicating valid right connection data,
// "lll" is a 3-bit offset index for the direction of the left connection
// "rrr" is a 3-bit offset index for the direction of the right connection

//NOTE: returned values in find_segment_starts.cl are in the form 0bSE0LRrrr where
// "S" is the start indicator flag,
//TODO: v swap this flag's sense so that overruns into unwritten pixels aren't possible
// "E" is end adjacent indicator flag, ie. cont_data still valid but next pixel coord won't be, doesn't handle start being next
// "L" is the left support indicator flag,
// "R" is occupancy/right continuation indicator flag, and	//NOTE: may deprecate this flag in favor of just the "E" flag
// "rrr" is the 3-bit offset direction index of the right continuation

#endif//LINK_MACROS_CL