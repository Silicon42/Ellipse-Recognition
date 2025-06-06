// returns the square of the distance of an int2 vector, doesn't protect against overflow
/*uint length2(int2 vec)
{
	vec *= vec;
	return vec.x + vec.y;
}
*/
float cross_2d_f(float2 a, float2 b)
{
	return a.x * b.y - a.y * b.x;
}

int cross_2d_i(int2 a, int2 b)
{
	return a.x * b.y - a.y * b.x;
}

int cross_2d_c(char2 a, char2 b)
{
	return a.x * b.y - a.y * b.x;
}

int dot_2d_i(int2 a, int2 b)
{
	return a.x * b.x + a.y * b.y;
}

float dot_2d_f(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}
/*
double dot_2d_d(double2 a, double2 b)
{
	return a.x * b.x + a.y * b.y;
}
*/
uint mag2_2d_i(int2 a)
{
	a *= a;
	return a.x + a.y;
}

float mag2_2d_f(float2 a)
{
	a *= a;
	return a.x + a.y;
}

float mag_2d_i(int2 a)
{
	a *= a;
	return sqrt((float)(a.x + a.y));
}

//unsafe for relatively large values,
// however I only use it for mid-point to end-point deflection of line checks which are all small
uchar mag2_2d_c(char2 a)
{
	char2 a2 = a * a;
	return a2.x + a2.y;
}

int2 perp_2d_i(int2 a)
{
	return (int2)(-a.y, a.x);
}

uint taxi_len_2d_i(int2 a)
{
	uint2 c = abs(a);
	return c.x + c.y;
}

float2 intersect_ab_cd(float2 a, float2 b, float2 c, float2 d)
{
	float2 ab, cd;
	ab = b - a;
	cd = d - c;
	return (cd * cross_2d_f(b, a) + ab * cross_2d_f(c, d)) / cross_2d_f(ab, cd);
}