// de-duplicates similar solutions via mean-shift clustering
// however instead of a hard cut-off threshold we use a weighted average that favors nearer points more
#include "ellipse_dedup_metric.cl_h"

kernel void ellipse_dedup_1(
	read_only image2d_t ff4_sol_coeffs_ACDE,
	read_only image2d_t ff1_sol_coeffs_B,
	write_only image2d_t ff4_cluster_ACDE,	// can't safely write directly back since reads need unmodified original data
	write_only image2d_t ff1_cluster_B)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	float4 this_acde = read_imagef(ff4_sol_coeffs_ACDE, indices);
	if(all(this_acde == 0))
		return;
	
	float this_b = read_imagef(ff1_sol_coeffs_B, indices).x;

	float4 acde_accum = 0;
	float b_accum = 0;
	
	const int bound = get_global_size(0);
	do	// while total motion above STILL_THRESH
	{
		for(int i = 0; i < bound; ++i)
		{
			float4 curr_acde = read_imagef(ff4_sol_coeffs_ACDE, (int2)(i, indices.y));
			if(all(curr_acde == 0))
				continue;
			
			float curr_b = read_imagef(ff1_sol_coeffs_B, (int2)(i, indices.y)).x;
			float4 diff_acde = curr_acde - this_acde;
			float diff_b = curr_b - this_b;
			float weight = 1 + weight_5D(diff_acde, diff_b);	// +1 to prevent divide by 0 from occurring
			
			acde_accum += diff_acde / weight;
			b_accum += diff_b / weight;
		}
		this_acde += acde_accum;
		this_b += b_accum;
		printf("%e ", weight_5D(acde_accum, b_accum));
	} while(weight_5D(acde_accum, b_accum) > STILL_THRESH);

	write_imagef(ff4_cluster_ACDE, indices, this_acde);
	write_imagef(ff1_cluster_B, indices, this_b);
}