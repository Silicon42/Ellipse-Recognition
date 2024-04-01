//TODO: fix defines and ifdefs so that h_dims and i_dims may be pre-defined and therefore avoid being calculated for each thread

//TODO: add a shared header for hough_lines so the defines aren't duplicated
#ifndef HOUGH_HEIGHT
#define HOUGH_HEIGHT	(h_dims.y)
#endif//HOUGH_HEIGHT

#ifdef  ANGLE_BITS
#define ANGLE_RES		(1<<ANGLE_BITS)
#else
#define ANGLE_RES		(h_dims.x)
#endif//ANGLE_BITS
// how much to multiply by to shift the most significant angle bit to the msb of a short
#define ANGLE_16_BITS		((1<<16)/ANGLE_RES)
//TODO: check if this actually calculates or if the compiler is smart enough
#define ANGLE_SCALE			(2*M_PI_F/ANGLE_RES)
#define DIVERGENCE_THRESH	(1<<(15-3))

// How much to divide the h_coord by to fit it into the 0 to 255 range
#define ANGLE_8_DIV		(ANGLE_RES/256)
/*
union vec2ToArr {
	float2 v;
	float a[2];
};*/

// [0] In	us1_src_image: 1 channel peaks from Hough (lines) accumulator (UINT16),
//				x-axis reperesents rotation 0 to 360, y-axis represents signed
//				distance along edge gradient direction to center of image.
// [1] Out	uc1_dst_image: 2 channel, 1 boolean of original image dimensions where
//				lines corresponding to the peaks in Hough space are drawn to, and
//				1 channel (INT8) representation of angle
__kernel void inv_hough_lines(read_only image2d_t us1_src_image, write_only image2d_t uc1_dst_image)
{
	// determine ouput coordinate for work item
	int2 h_coords = (int2)(get_global_id(0), get_global_id(1));
	short is_peak = read_imageui(us1_src_image, h_coords).x;
	if(!is_peak)
		return;

//TODO: ifdefs here
	int2 h_dims = get_image_dim(us1_src_image);
	int2 i_dims = get_image_dim(uc1_dst_image);

	//FIXME: this probably isn't rounding the right way, come back and fix this when not tired
	char angle = h_coords.x / ANGLE_8_DIV;
	// convert from normal (polar) to normal (cartesian) form
	union vec2ToArr grad;
	grad.a[1] = sincos(h_coords.x * ANGLE_SCALE, grad.a);
	// account for center offset parameterization choice while we're at it
	float2 normal = mad(grad.v, h_coords.y - HOUGH_HEIGHT/2, convert_float2(i_dims / 2));

	// direction to step along the line is 90 degree to gradient/normal
	float2 step = grad.v.yx * (float2)(-1, 1)*0.5;// * 0.707f;

	// convert coords to a signed 16 bit fixed precision angle representation that wraps properly
	short h_angle16 = h_coords.x * ANGLE_16_BITS;
	float2 bounds = convert_float2(i_dims - 1);	// no indexing past the last pixel
	float accum[2] = {0};
	float2 i_coords = normal;

	bool twice = false;
	do
	{
		// check if coords are within image bounds, otherwise it will sample garbage
		while(all((0.0f <= i_coords) & (i_coords <= bounds)))
		{
			// mark that a line was found and its fixed precision angle at the current point
			write_imageui(uc1_dst_image, convert_int2_rtz(i_coords), (uint4)(is_peak, angle, 0, -1));
			i_coords += step;
		}
		step *= -1;
		i_coords = normal + step;
	} while(!twice++);
}