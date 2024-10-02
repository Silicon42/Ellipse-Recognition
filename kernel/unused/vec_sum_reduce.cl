#ifndef VEC_SUM_REDUCE_CL
#define VEC_SUM_REDUCE_CL

char sum_reduce(char8 vec)
{
	vec.lo += vec.hi;
	vec.xy += vec.zw;
	return vec.x + vec.y;
}

#endif//VEC_SUM_REDUCE_CL