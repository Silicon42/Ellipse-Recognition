#include <stdio.h>
#include <math.h>
#define NUM_BINS		1024
#define ANGULAR_STEP	(2*M_PI / NUM_BINS)
int main()
{
	printf("255\t");
	for(int i = 1; i < NUM_BINS; ++i)
		printf("%i\t", (unsigned char)(128 *(1 + cos(i * ANGULAR_STEP))));
	printf("\n");
}