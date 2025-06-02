// generates the possible groupings of 4 edges out of 8 and verifies correctness of loop iteration for arc clique finding
#include <stdio.h>
#define MAX_CANDIDATES 8

int main()
{
	unsigned char edge_sets2[28] = {0};

	// 2 clique processing
	for(int i = 0, k = 0; i < MAX_CANDIDATES; ++i)
	{
		unsigned char processed_1 = 1 << i;
		for(int j = i+1; j < MAX_CANDIDATES; ++j, ++k)
		{
			edge_sets2[k] = processed_1 | (1 << j);
			printf("%02X, ", edge_sets2[k]);
		}
	}
	puts("\n");

	//Just for validating order
	// 3 clique processing, special because it doesn't require storing to edge_sets[]
	// and must be formed with entries with at least one shared node, plus iterated 
	// such that we preferrably avoid duplicate computations
	for(int i = 0, i_step = MAX_CANDIDATES-1, i_thresh = i_step; i_step > 1; i_thresh += i_step)
	{
		unsigned char processed_1 = 1 << (MAX_CANDIDATES-1 - i_step);
	//	printf("%02X, ", processed_1);
		--i_step;
		for(int j = i_thresh, j_step = i_step, j_thresh = i_thresh + i_step; i < i_thresh; j_thresh += --j_step, ++i)
		{
			unsigned char processed_2 = 1 << (MAX_CANDIDATES-1 - j_step);
		//	printf("%02X, ", processed_2);
			for(; j < j_thresh; ++j)
			{
				unsigned char processed = processed_1 | processed_2 | (1 << (MAX_CANDIDATES + j - j_thresh));	//TODO: this might be faster with a LUT
			//	printf("%02X|%02X, ", edge_sets2[i], edge_sets2[j]);
			//	if(__builtin_popcount(edge_sets2[i] | edge_sets2[j]) != 3)
				
				if(processed != (edge_sets2[i] | edge_sets2[j]))
					printf("%02X|%02X!=%02X!|%02X!|%02X!, ", edge_sets2[i], edge_sets2[j], processed_1, processed_2, (1 << (MAX_CANDIDATES + j - j_thresh)));
		}
		}
	}

	// 4 clique processing
	for(int i = 0, k = 0, i_step = MAX_CANDIDATES-1, i_thresh = i_step; i_step > 2; i_thresh += i_step)
	{
		--i_step;
		for(int j_step = i_step, j0 = i_thresh; i < i_thresh; ++i)
		{
			j0 += j_step;
			--j_step;
			for(int j = j0; j < 28; ++j, ++k)
			{
				printf("0x%02X, ", edge_sets2[i] | edge_sets2[j]);
				if(__builtin_popcount(edge_sets2[i] | edge_sets2[j]) != 4)
					puts("Oh no!");
		}
		}
	}

}