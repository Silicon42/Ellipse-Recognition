#include "ellipse_dedup_metric.cl_h"

// generic "distance" function for determining how to weight a 5D point's effect on the local "mean" position
float weight_5D(float4 dim0123, float dim4)
{
	dim0123 *= dim0123;
	return dim0123.x + dim0123.y + dim0123.z + dim0123.w + dim4*dim4;
}