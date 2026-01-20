#include "conic_solve.cl_h"
#include "math_helpers.cl_h"

// computes A = (B^T)B for B width 5, height m, used as part of calculating pseudo-inverse
void selfTransposeProduct(int m, float B[][5], float A[15])
{
	// A's contents are assumed to be initialized to 0
	for(int row = m-1; row >= 0; --row)
		for(int i = 4; i >= 0; --i)
			for(int j = i; j >= 0; --j)
				A[TRI_INDEX(i,j)] += B[row][i]*B[row][j];
}

void cholesky_inv_sym_5(float A[15])
{
	// GCC only unrolls deepest loops but for this unrolling is both faster
	// and results in a shorter program
	#pragma GCC unroll 9
	// for each row
	__attribute__((opencl_unroll_hint(4)))
	for(int i = 1; i < 5; ++i)
	{	// calculate the D**-1 coeff, not stored as D coeff as a speed/accuracy 
		// tradeoff, doing this only uses 5 reciprocals + 15 multiplies and doing
		// the division normally uses 15 divides, if we assume a divide is 2x the
		// cycle cost of a multiply (or worse), the total cost is 5*2 + 15 = 25 
		// multiplies this way vs 15*2 = 30 multiplies when using as part of a 
		// solver for systems of linear equations, with 5 of the operations going
		// toward applying the diagonal to a column vector
		A[TRI_INDEX(i,-1)] = 1 / A[TRI_INDEX(i,-1)];
		// for each element in the current row before the diagonal
		for(int j = 0; j < i; ++j)
		{	// calculate the -L coefficient, negated b/c it's only ever needed as a negative
			// stored in temp variable so as to not overwrite the cell before using its S element
			float L_temp = -A[TRI_INDEX(i,j)] * A[TRI_INDEX(j,j)];
			for(int k = i; k < 5; ++k)
				A[TRI_INDEX(k,i)] += L_temp * A[TRI_INDEX(k,j)];

			A[TRI_INDEX(i,j)] = L_temp;
		}
	}
	// get reciprocal of the last D coeff to convert to D**-1 coeff for consistency
	// could leave it and manually have a separate divison to slightly mitigate
	// the compounding accuracy loss, however any vectorizing wouldn't be able 
	// to handle that
	A[TRI_INDEX(4,4)] = 1 / A[TRI_INDEX(4,4)];

	// now that the L coefficients are fully calculated, they can be inverted by
	// row operations of subtracting multiples of lower rows. Since this would
	// zero out the element at that position, we can use the same element to 
	// store the resulting value that corresponds to the right side of the
	// augmented matrix

	#pragma GCC unroll 9
	// for each row, bottom first
	for(int i = 3; i > 0; --i)
		for(int j = i - 1; j >= 0; --j)	// for each element below the diagonal, right-most first
			for(int k = i + 1; k < 5; ++k)	// for each element below [i][j]
				A[TRI_INDEX(k,j)] += A[TRI_INDEX(i,j)] * A[TRI_INDEX(k,i)];

}

// Takes a packed, array of floats in A representing the coefficients of the 
// matrices L and D and vector b as produced by Cholesky LDL^-1 decomposition 
// and solves for the 5 coefficients of a best fit, general conic equation and 
// returns them in b
void solveConic(float A[15], float b[5])
{
	// expand b coefficients
	b[0] = A[10];	// Ex
	b[1] = A[11];	// Dy
	b[2] = A[0];	// Ax^2
	b[3] = A[2];	// Cy^2
	b[4] = A[1];	// Bxy

	// copy duplicate A coefficients into position
	A[8] = A[14];
	A[10] = A[4];
	A[11] = A[6];

	cholesky_inv_sym_5(A);

	// (L^-1)^T(D(L^-1 * b))

	// b = L^-1 * b
	for(int i = 4; i > 0; --i)
		for(int j = i-1; j >= 0; --j)
			b[i] += A[TRI_INDEX(i,j)] * b[j];

	// b = D * b
	for(int i = 4; i >= 0; --i)
		b[i] *= A[TRI_INDEX(i,i)];
	
	// b = (L^-1)^T * b
	for(int i = 0; i < 4; ++i)
		for(int j = i+1; j < 5; ++j)
			b[i] += A[TRI_INDEX(j,i)] * b[j];
}

// returns the sum of the distances from an ellipse foci and a point
inline float get_ellipse_dist(const float4 foci, const float2 point)
{
	return fast_distance(foci.lo, point) + fast_distance(foci.hi, point);
}

// returns the difference of the distances from a hyperbola foci and a point
inline float get_hyperbola_dist(const float4 foci, const float2 point)
{
	return fabs(fast_distance(foci.lo, point) - fast_distance(foci.hi, point));
}

// returns the absolute difference between a conic's major axis length vs a point and it's foci
// this can be used to determine if a point is close to the conic's boundary
float get_conic_deviation(FociMajor const * const M, const float2 point)
{
	float dist = (M->major >= 0) ? get_ellipse_dist(M->foci, point) : -get_hyperbola_dist(M->foci, point);
	return fabs(M->major - dist);
}

// following few functions are somewhat sensitive to fused multiply-add and will give wrong answers if they use them in some cases
// so to prevent that, the optimizations must be disabled for them
#pragma OPENCL FP_CONTRACT OFF

// gets a consistently scaled rough approximation of an ellipse's perimeter
// used for ranking how much of the calculated ellipse a set of arc segments 
// actually accounts for as calculated by the sum of their line segment lengths
// divided by the coverage divisor
//NOTE: currently uses the Kummer infinite sum approximation with very few terms used
// (https://en.wikipedia.org/wiki/Perimeter_of_an_ellipse#Infinite_sums)
//NOTE: scaling factor: is currently a factor of pi smaller than the perimeter (at time of writing 6/12/2025) 
float get_ellipse_coverage_divisor(float gen[5])
{
	float b = gen[4];
	if(b <= 0)	// hyperbola's can't have coverage computed
		return -1;
	float b2 = b*b;
	float t2 = 4*gen[2]*gen[3] - b2;	// t^2 = 4ac - b^2
	if(!isfinite(t2) || t2 <= 0)		// only continue if t^2 is positive finite, otherwise not an ellipse and perimeter is infinite
		return -1;

	float det_M, ac_diff2, ac_b_len;
	ac_diff2 = gen[2] - gen[3];
	ac_diff2 *= ac_diff2;	// (a-c)^2
	float2 ed, ac, temp_f2;
	ed = (float2)(gen[1], gen[0]);
	ac = (float2)(gen[2], gen[3]);
	temp_f2 = ed * ed * ac;			// [ae^2, cd^2]
	det_M = -2*(t2 - b * ed.x * ed.y + temp_f2.x + temp_f2.y);	// det(M) = -2*(t^2 + ae^2 - bde + cd^2)	//NOTE: -2 scalar ommitted as not relevant here
	ac_b_len = sqrt(ac_diff2 + b2);

	// semi-major and semi-minor axis lengths but with a sqrt(det(M)/t^2) scale factor deffered for calculation simplification reasons
	float semimajor = -1 / (ac.x + ac.y + ac_b_len);		// semimajor^2 / det(M)/t^2 = -1 / (a + c + sqrt((a-c)^2 + b^2)
	float semiminor = sqrt(semimajor - 2 * ac_b_len / t2);	// semiminor / sqrt(det(M)/t^2)
	semimajor = sqrt(semimajor);	//semimajor / sqrt(det(M)/t^2)

	// real approx is pi*(maj + min)*(1 + h/4 + h^2/64 + h^3/256 + ...) where h = ((maj - min)/(maj + min))^2
	// What's actually calculated here (at time of writing 5/20/2025) is (a + b)(4 + h)
	// deferred scale factors cancel here for h but not for axis_sum
	float axis_sum = semimajor + semiminor;	// (semimajor + semiminor)/sqrt(det(M)/t^2)
	float h = (semimajor - semiminor) / axis_sum;
	h *= h;
	return sqrt(det_M / t2) * axis_sum * (4 + h);	// reintroduce sqrt(det(M)/t^2) scale factor that was omitted in semi-major and semi-minor calc
//	printf("%f	", ret);
//	return ret;
}

// Converts an ellipse in general conic form to foci-major form
// if not an ellipse, returns non-positive distance
//NOTE: derived from here: https://math.stackexchange.com/questions/44391/foci-of-a-general-conic-equation
void convertConicGeneralToFociMajor(Conic * const M, bool isntThruOrigin)
{
	float* gen = M->general;
	FociMajor* fm = &M->fm;
	float b = gen[4];
//printf("%v4f deac? %f b\n", fm->foci, b);
	float t2 = 4*gen[2]*gen[3] - b*b;	// t^2 = 4ac - b^2
	//only bother computing foci for ellipse or hyperbola candidates, not parabolas due to divide by 0, or degenerate conics
	if(!isfinite(t2) || t2 == 0)
	{
		fm->major = NAN;
		return;
	}

	float det_M, ac_diff, ac_b_len;
	ac_diff = gen[2] - gen[3];
	float2 ed, ac, rs, temp_f2;
	ed = (float2)(gen[1], gen[0]);
	ac = (float2)(gen[2], gen[3]);
	rs = b * ed;				// b[e, d]
	det_M = rs.x * ed.y;		// bde
	if(isntThruOrigin)			// f == 0 if thru origin
		det_M -= t2;			// bde + 2tf, where f == -1
//printf("%f bde (+2tf)\n", det_M);
	temp_f2 = ed * ac;			// [ae, cd]
	rs -= 2 * temp_f2.yx;		// b[e, d] - 2[cd, ae]
//printf("%v2f rs = b[e, d] - 2[cd, ae]\n", rs);
	temp_f2 *= -ed;				// [-ae^2, -cd^2]
//printf("%v2f temp_f2\n", temp_f2);
	det_M = 2*(det_M + temp_f2.x + temp_f2.y);	// det(M) = 2*(2tf - ae^2 + bde - cd^2), f == 0 if thru origin, else f == -1
//printf("%f det_M\n", det_M);
	ac_b_len = hypot(ac_diff, b);
	int signbit_t2 = signbit(t2);
	int sign_M = signbit(det_M) ? -1 : 1;	// extract sign for sign dependent logic
	det_M = fabs(det_M);		// prevent sqrt of negative from occurring
//printf("%f ac_b_len %f ac_diff %f 2t\n", ac_b_len, ac_diff, t2);
//printf("%v2f rel center\n", rs/t2);
	// [1, sign(b)] * sqrt(2 * det(M) * (hypot(a-c, b) + [a-c, c-a]))
	temp_f2 = (float2)(1, signbit_t2 ? 1 : -1) * sqrt(det_M * (ac_b_len + (float2)(sign_M, -sign_M) * ac_diff));
	fm->major = 2*sqrt(fabs(det_M * (ac.x + ac.y + ((signbit(b)^signbit_t2) ? -1 : 1)*ac_b_len))) / t2;
	fm->foci.lo = rs + temp_f2;
	fm->foci.hi = rs - temp_f2;
	fm->foci /= t2;
//printf("%v4f foci %f major\n", fm->foci, fm->major);
}


// calculates a conic through 5 points where 1 point is (0,0) and the rest are relative to it
// returns the foci coordinates, distance from foci to edge is implied
// if the conic through 5 points would not be an ellipse, returns a non-positive distance
void conic_from_hist(private const int2 diffs[4], private const int cross_prods[4], Conic * const conic)
{	//TODO: see how to mitigate rounding errors better
	float2 ca, ed;//, rs, temp_f2;
	float b, u, v;// det_M, inv_2t, ac_diff, ac_b_len;
	int2 temp_i2;

	// Fix to prevent exponent overflow from too many multiplication steps by pre-scaling the u and v values
	// technically it might be safer to divide by the avg exponent between the max and non-zero-min of the coefficients,
	// but dividing by a constant power of 2 is faster and should work in most cases, especially if resolution is kept
	// to reasonable values (ie roughly <= 4069)
	//FIXME: max guaranteed safe divisor with -cl-denorms-are-zero set is 2147483648 (2^31), need to add defines that take that into account
	u =  (cross_prods[1] * cross_prods[3]);//2147483648.0f;// / 137438953472.0f;	// bias exponent by dividing by 2^37, max safe value without losing fine resolution
	v = -(cross_prods[0] * cross_prods[2]);//2147483648.0f;// / 137438953472.0f;	// compiler should hopefully optimize this to simple exponent setting since it's a power of 2
//	printf("%f	%f	u,v\n", u, v);
	ca = u * convert_float2(diffs[0] * diffs[2]) + v * convert_float2(diffs[1] * diffs[3]);
	temp_i2 = diffs[0] * diffs[2].yx;
	b  = u * (float)(temp_i2.x + temp_i2.y);
	temp_i2 = diffs[1] * diffs[3].yx;
	b += v * (float)(temp_i2.x + temp_i2.y);
	b = -b;

	ed  = u * (cross_prods[0] * convert_float2(diffs[2]) + cross_prods[2] * convert_float2(diffs[0]));
//printf("%v2f	%i %i	%f\n", ed, cross_prods[0], cross_prods[2], u);
	ed += v * (cross_prods[1] * convert_float2(diffs[3]) + cross_prods[3] * convert_float2(diffs[1]));
	ed.x = -ed.x;
	if(b != 0)
	{
		ed /= b;
		ca /= b;
		b = 1;
	}
	//TODO: change conic type so that this assignment makes more sense, as this is technically a general type conic at this point
	conic->fm.foci = (float4)(ed.yx, ca.yx);	//FIXME: SOMEWHERE I got the ordering of the e and d components mixed up but as it is now it currently works so will be left as is
	conic->fm.major = b;
	convertConicGeneralToFociMajor(conic, false);
//printf("%v4f foci\n", conic->fm.foci);
	return;
}

#pragma OPENCL FP_CONTRACT DEFAULT

// calculate the coefficient components for this point to the square matrix
// done in long int math to prevent precision loss, safe from overflow as long as
// no more than 256 points are added this way for max x and y coords < 16384 (2^14),
// safe limit is higher if max coords are less than that, considering the longest
// chain I've seen so far was ~30, I'm not even going to check
// Assumes x and y coords are in the range of 0 to 16383
ulong16 getPointCoeffs(int2 p)
{
	// x2 and y2 are stored in ulongs so promotion occurs before operations in the return statement
	// could be done just as well with casting but is more readable this way
	ulong x2 = p.x * p.x;
	ulong y2 = p.y * p.y;
	uint xy = p.x * p.y;
	//TODO: this could probably be done in a more efficient order, also 
	// the ordering might not be ideal for later conversion to packed symmetric 
	// form from stored coefficients in terms of accuracy loss
	return (ulong16)(
		x2,		xy,		y2,		x2*p.x,	//	x^2		xy		y^2		x^3
		x2*p.y,	x2*x2,	p.x*y2,	p.y*y2,	//	x^2y	x^4		xy^2	y^3
		0,		y2*y2,	p.x,	p.y,	//	perim.	y^4		x		y
		x2*xy,	xy*y2,	x2*y2,	1);		//	x^3y	x^y3	x^2y^2	2*count
}

void readPreSolveCoeffs(read_only image2d_t ff4_pre_solve_coeffs, int2 coords, float16* ret)
{
	coords *= 2;
	(*ret).hi.hi = read_imagef(ff4_pre_solve_coeffs, coords + 1);
	(*ret).hi.lo = read_imagef(ff4_pre_solve_coeffs, coords + (int2)(0,1));
	(*ret).lo.hi = read_imagef(ff4_pre_solve_coeffs, coords + (int2)(1,0));
	(*ret).lo.lo = read_imagef(ff4_pre_solve_coeffs, coords);
}
