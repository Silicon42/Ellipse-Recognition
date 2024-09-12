#ifndef MATH_HELPERS_CL
#define MATH_HELPERS_CL

// returns the square of the distance of an int2 vector, doesn't protect against overflow
/*inline uint length2(int2 vec)
{
	vec *= vec;
	return vec.x + vec.y;
}
*/
inline float cross_2d(float2 a, float2 b)
{
	return a.x * b.y - a.y * b.x;
}

inline int cross_2d_i(int2 a, int2 b)
{
	return a.x * b.y - a.y * b.x;
}

inline int dot_2d_i(int2 a, int2 b)
{
	return a.x * b.x + a.y * b.y;
}

inline uint mag2_2d_i(int2 a)
{
	int2 a2 = a * a;
	return a2.x + a2.y;
}

inline int2 perp_2d_i(int2 a)
{
	return (int2)(-a.y, a.x);
}

#endif//MATH_HELPERS_CL