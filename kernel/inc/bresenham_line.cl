#ifndef BRESENHAM_LINE_CL
#define BRESENHAM_LINE_CL

#include "cast_helpers.cl"

void draw_line(int2 a, int2 b, uint4 color, write_only image2d_t uc4_out_image)
{
	union l_conv bounds;
	bounds.i = get_image_dim(uc4_out_image);
	
	uint2 abs_d = abs_diff(a, b);
	uchar swap_xy = abs_d.y > abs_d.x;
	if(swap_xy)
	{
		a = a.yx;
		b = b.yx;
		bounds.i = bounds.i.yx;
	}

	if(a.x > b.x)	//swap start and end
	{
		int2 temp = a;
		a = b;
		b = temp;
	}

	int2 d = b - a;
	
	int dir = (d.y >= 0) ? 1 : -1;	// get stepping direction (up or down)
	d.y *= dir;

	union l_conv c;
	c.i = a;	// current position starts at a
	int p = 2*d.y - d.x;	// decision parameter
	for(; c.i.x <= b.x; ++c.i.x)
	{
		if(all(c.ui < bounds.ui))	// check in bounds before drawing
		{
			if(swap_xy)
				write_imageui(uc4_out_image, c.i.yx, color);
			else
				write_imageui(uc4_out_image, c.i, color);
		}
		if(p >= 0)
		{
			c.i.y += dir;
			p -= 2*d.x;
		}
		p += 2*d.y;
	}
}

#endif//BRESENHAM_LINE_CL