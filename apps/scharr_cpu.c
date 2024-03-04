#include <stdio.h>
#include "stb_image.h"
#include "stb_image_write.h"

int main()
{
	// Read pixel data
	int dims[3];
	unsigned char* data = stbi_load("input.png", &dims[0], &dims[1], &dims[2], 1);
	if(!data)
	{
		perror("Couldn't open input image");
		exit(1);
	}
	unsigned char* out_data = (unsigned char*)malloc(dims[0] * dims[1] * 3);

	for(int x = 0; x < dims[0]; ++x)
	{
		for(int y = 0; y < dims[1]; ++y)
		{
			// itermediate sums for horizontal/vertical scharr shared operations
			float diag1, diag2, gradx, grady;

			//NOTE: these indexing operations are NOT safe practice since this is just for debugging and performance comparison with OpenMP
			// calculate intermediate sums from raw pixels
			diag1 = data[(x-1) + (y-1)*dims[0]];
			diag1 -= data[(x+1) + (y+1)*dims[0]];
			diag2 = data[(x-1) + (y+1)*dims[0]];
			diag2 -= data[(x+1) + (y-1)*dims[0]];
			gradx = data[(x-1) + (y+0)*dims[0]];
			gradx -= data[(x+1) + (y+0)*dims[0]];
			grady = data[(x+0) + (y-1)*dims[0]];
			grady -= data[(x+0) + (y+1)*dims[0]];

			gradx = gradx * 3.44680851 + diag1 + diag2;
			grady = grady * 3.44680851 + diag1 - diag2;
			
			out_data[(x + y*dims[0])*3] = gradx * 0.091796875 + 128;
			out_data[(x + y*dims[0])*3 + 1] = grady * 0.091796875 + 128;
			out_data[(x + y*dims[0])*3 + 2] = 128;
		}
	}

	free(data);
	stbi_write_png("output.png", dims[0], dims[1], 3, out_data, dims[0]*3);
	free(out_data);
}