#ifndef MIDPOINT_CIRCLE_CL
#define MIDPOINT_CIRCLE_CL

#include "math_helpers.cl"

// uses midpoint circle algorithm to draw either a face or corner centric circle (nearest to true center)
void draw_circle(float2 center, float r, write_only image2d_t ui4_output)
{
	int2 c, c_corner;
	c_corner = convert_int2(center);
	c = convert_int2(center - 0.5f);
	// if distance to nearest pixel center is greater than distance to nearest pixel corner,
	// use corner centric calculations for draw location
	char is_corner = fast_distance(convert_float2(c), center) > fast_distance(convert_float2(c_corner), center);
	if(is_corner)
		c = c_corner;

	
	int2 a = (int2)(r,0);
	int p = r;
	//while(a.x >= a.y)
}

#endif//MIDPOINT_CIRCLE_CL