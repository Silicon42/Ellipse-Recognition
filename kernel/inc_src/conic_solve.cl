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
	b[0] = A[10];
	b[1] = A[11];
	b[2] = A[0];
	b[3] = A[2];
	b[4] = A[1];

	// copy duplicate A coefficients into position
	A[8] = A[14];
	A[10] = A[4];
	A[11] = A[6];

	cholesky_inv_sym_5(A);

	// (L^-1)^T(D(L^-1 * b))

	// b = L^-1 * b
	for(int i = 4; i > 0; --i)
		for(int j = i-1; j >= 0; --i)
			b[i] += A[TRI_INDEX(i,j)] * b[j];

	// b = D * b
	for(int i = 4; i >= 0; --i)
		b[i] *= A[TRI_INDEX(i,i)];
	
	// b = (L^-1)^T * b
	for(int i = 0; i < 4; ++i)
		for(int j = i+1; j < 5; ++j)
			b[i] += A[TRI_INDEX(j,i)] * b[j];
}

// Converts an ellipse in general conic form to foci-distance form
// returns the foci on the first 4 elements of b and distance on the 5th
// if not an ellipse, returns negative distance
void convertGeneralConicToFociDistEllipse(float b[5])
{
	float t2 = 4*b[2]*b[4] - b[3]*b[3];
	if(t2 <= 0)
	{
		b[4] = -1;
		return;
	}

	float2 rs = (float2)(b[1], b[0]) * b[3];
}


// calculates a ellipse through 5 points where 1 point is (0,0) and the rest are relative to it
// returns the foci coordinates, distance from foci to edge is implied
// if the conic through 5 points would not be an ellipse, returns NaN
#pragma OPENCL FP_CONTRACT OFF
float4 ellipse_from_hist(private const int2 diffs[4], private const int cross_prods[4])
{	//TODO: see how to mitigate rounding errors better
//if(all(diffs[0]==(int2)(75,-27)))
//	printf("%i	%i	%i	%i\n", cross_prods[0],cross_prods[1],cross_prods[2],cross_prods[3]);
//	printf("%v2i	%v2i	%v2i	%v2i\n", diffs[0],diffs[1],diffs[2],diffs[3]);
	float4 foci;
	float2 ca, ed, rs, temp_f2;
	float b, temp_f, inv_2t, ac_diff;
	float u, v;
	int2 temp_i2;

	// Fix to prevent exponent overflow from too many multiplication steps by pre-scaling the u and v values
	// technically it might be safer to divide by the avg exponent between the max and non-zero-min of the coefficients,
	// but dividing by a constant power of 2 is faster and should work in most cases, especially if resolution is kept
	// to reasonable values (ie roughly <= 4069)
	//FIXME: max guaranteed safe divisor with -cl-denorms-are-zero set is 2147483648 (2^31), need to add defines that take that into account
	u =  (cross_prods[1] * cross_prods[3]) / 137438953472.0f;	// bias exponent by dividing by 2^37, max safe value without losing fine resolution
	v = -(cross_prods[0] * cross_prods[2]) / 137438953472.0f;	// compiler should hopefully optimize this to simple exponent setting since it's a power of 2

	ca = u * convert_float2(diffs[0] * diffs[2]) + v * convert_float2(diffs[1] * diffs[3]);
	temp_i2 = diffs[0] * diffs[2].yx;
	b = -u * (float)(temp_i2.x + temp_i2.y);
	temp_i2 = diffs[1] * diffs[3].yx;
	b -= v * (float)(temp_i2.x + temp_i2.y);
//ca = (float2)(-40,-33);
//b=-24;
	inv_2t = 4 * ca.x * ca.y - b * b;
if(all(diffs[0]==(int2)(75,-27)))
printf("%A	", inv_2t);
	//only bother computing foci for ellipse candidates, not parabolas or hyperbolas
	if(inv_2t <= 0)
		return NAN;
	
	inv_2t = 1 / inv_2t;

	ed = u * (cross_prods[0] * convert_float2(diffs[2]) + cross_prods[2] * convert_float2(diffs[0]))\
		+v * (cross_prods[1] * convert_float2(diffs[3]) + cross_prods[3] * convert_float2(diffs[1]));
if(all(diffs[0]==(int2)(75,-27)))
printf("%v2A	", ca);
//ed=(float2)(168);
	char negate = all(ca < 0) ? -1:1;	// this is to prevent the temp_f value from going negative because the square root can't handle that
	b *= negate;
	ed *= (float2)(-negate, negate);
	ca *= negate;
//if(negate < 0)
//	printf("n");

	rs = b * ed;			//b[e, d]
	temp_f = -rs.x * ed.y;	//-bde
	temp_f2 = ca * ed.yx;	//[cd, ae]
	rs -= 2 * temp_f2;		//b[e, d] - 2[cd, ae]
	ac_diff = ca.y - ca.x;	//a-c

	temp_f = 2 * (temp_f + dot_2d_f(temp_f2, ed.yx));	//2(ae^2 - bde + cd^2)
	temp_f2 = sqrt(temp_f * (hypot(ac_diff, b) + (float2)(-ac_diff, ac_diff)));
if(any(isnan(temp_f2)))
	printf("X");

	// due to sqrt of complex value, x and y components are either same sign if b > 0 or opposite sign if b < 0
	if(b > 0)
		temp_f2.y *= -1;

	foci.lo = rs - temp_f2;
	foci.hi = rs + temp_f2;
	foci *= inv_2t;
//printf("%v4f ]\n", foci);
	
	return convert_float4(foci);
}
#pragma OPENCL FP_CONTRACT DEFAULT

// adds the coefficient components as calculated for this point to the square matrix
// done in long int math to prevent precision loss, safe from overflow as long as
// no more than 256 points are added this way for max x and y coords < 16384 (2^14),
// safe limit is higher if max coords are less than that, considering the longest
// chain I've seen so far was ~30, I'm not even going to check
void addPointCoeffs(ulong4 coeffs[4], int2 p)
{
	ulong x2 = p.x * p.x;
	ulong y2 = p.y * p.y;
	ulong xy = p.x * p.y;
	//TODO: this could probably be done in a more efficient order, also 
	// the ordering might not be ideal for later conversion to packed symmetric 
	// form from stored coefficients in terms of accuracy loss
	coeffs[0] += (ulong4)(x2, xy, y2, x2*p.x);
	coeffs[1] += (ulong4)(x2*p.y, x2*x2, p.x*y2, p.y*y2);
	coeffs[2] += (ulong4)(1, y2*y2, p.x, p.y);
	coeffs[3] += (ulong4)(x2*xy, xy*y2, x2*y2, 0);
}

inline float get_ellipse_dist(const float4 foci)
{
	return fast_length(foci.lo) + fast_length(foci.hi);
}

inline char is_near_ellipse_edge(const float4 foci, const float dist, const float2 point)
{
	return fabs(dist - (fast_distance(point, foci.lo) + fast_distance(point, foci.hi))) < 2;
}