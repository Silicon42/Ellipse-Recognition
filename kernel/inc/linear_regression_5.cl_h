#define TRI_INDEX(i,j)	(((int)(i)*((int)(i) + 1))/2 + (int)(j))

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

// Takes a packed, 14 coefficient representing the self-transpose-product of the
// input points, in A[0-3, 8-15], and solves for the least squares fit of conic
// in linear form.
// The process to do so first, expands it to packed symmetric form, initializes 
// the b output vector,
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