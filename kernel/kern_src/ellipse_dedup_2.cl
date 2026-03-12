// de-duplicates similar solutions via mean-shift clustering
// part 2 does the association of points that ended up in the same group
#include "ellipse_dedup_metric.cl_h"
#include "cast_helpers.cl_h"

#define MAX_GROUP_SIZE	16

kernel void ellipse_dedup_2(
	read_only image2d_t ii4_sparse_adj_matrix,
	read_only image2d_t uc1_adj_consensus,
	read_only image2d_t ff4_cluster_ACDE,
	read_only image2d_t ff1_cluster_B,
	write_only image2d_t ii4_group_list)
{
	int2 indices = (int2)(get_global_id(0), get_global_id(1));
	float4 this_acde = read_imagef(ff4_cluster_ACDE, indices);
	if(all(this_acde == 0))
		return;
	
	float this_b = read_imagef(ff1_cluster_B, indices).x;
	
	const int bound = get_global_size(0);
	int group_size = 0;
	int lowest_member = indices.x;
	union s16_conv group_list = {.s = -1};
	for(int i = 0; i < bound; ++i)
	{
		float4 curr_acde = read_imagef(ff4_cluster_ACDE, (int2)(i, indices.y));
		if(all(curr_acde == 0))
			continue;
		
		float curr_b = read_imagef(ff1_cluster_B, (int2)(i, indices.y)).x;
		float4 diff_acde = curr_acde - this_acde;
		float diff_b = curr_b - this_b;
		float weight = weight_5D(diff_acde, diff_b);
		
		if(weight < 4*STILL_THRESH)
		{
			if(group_size == MAX_GROUP_SIZE)
			{
//				printf("TOO MANY MEMBERS\n");
				continue;
			}
			group_list.a[group_size] = i;
			if(i < lowest_member)
			{
				lowest_member = i;
				//TODO: commented out until threshold is deemed safe for general use
				//return;	// only the lowest index member may write
			}
			group_size++;
		}
		// temporary logic for tuning the threshold, if there are points that trigger this there could be groups getting missed
		// because they expect another work item to do the write, which itself is not writing because it thinks it's part of another group
		//currently the threshold is in euclidean distance squared, so this corresponds to being between 2 and 3 times the threshold away
		else
		{
			if(weight < 9*STILL_THRESH)
				printf("DANGER ZONE\n");
		}
	}

	// temporary until threshold is deemed safe for general use
	if(indices.x != lowest_member)
		return;

	int expanded_size = group_size;
	printf("%v16hi	@%i,%i\n", group_list.s, indices.x, indices.y);
	for(int i = 0; i < group_size; ++i)
	{
		union s8_conv adj_list = {.i = read_imagei(ii4_sparse_adj_matrix, indices)};
		uchar consensus = read_imageui(uc1_adj_consensus, indices).x;
		
		for(int j = 0; j < 8; ++j)
		{
			if((consensus & (1<<j)) && !any(group_list.s == adj_list.a[j]))
			{
				if(expanded_size == MAX_GROUP_SIZE)
				{
					printf("COULDN'T ADD MEMBER, FULL\n");
					continue;
				}
				group_list.a[expanded_size] = adj_list.a[j];
				++expanded_size;
			}
		}
	}
	write_imagei(ii4_group_list, indices, group_list.i.lo);
	write_imagei(ii4_group_list, (int2)(indices.x, indices.y | 2), group_list.i.hi);
}