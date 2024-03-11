__kernel void test(read_only image2d_t src_image)
{
	float8 x, y, res;
	x=(float8)(1, 1/M_SQRT2_F, 0, -1/M_SQRT2_F, -1, -1/M_SQRT2_F, 0, 1/M_SQRT2_F);
	y=(float8)(0, 1/M_SQRT2_F, 1, 1/M_SQRT2_F, 0, -1/M_SQRT2_F, -1, -1/M_SQRT2_F);
	res = atan2pi(y,x);
	printf("%f, %f, %f, %f, %f, %f, %f, %f\n", res.s0, res.s1, res.s2, res.s3, res.s4, res.s5, res.s6, res.s7);
}