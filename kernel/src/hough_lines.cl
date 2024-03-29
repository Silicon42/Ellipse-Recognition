//const sampler_t lin_samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;
/*
// Following defines need to be defined based on the input/output buffers at compile time
#ifndef IMG_WIDTH
#define IMG_WIDTH 1280
#endif//IMG_WIDTH

#ifndef IMG_HEIGHT
#define IMG_HEIGHT 720
#endif//IMG_HEIGHT

#define IMG_SIZE (IMG_WIDTH, IMG_WIDTH)
#define IMG_CENTER (IMG_SIZE>>1)
*/
#ifndef HOUGH_HEIGHT
#define HOUGH_HEIGHT		(h_dims.y)
#endif//HOUGH_HEIGHT
//must be pre-defined to avoid calculation on device
// how many bits of resolution for the Hough transforms angle dimension, max of 16 theoretical
// suggested 8 to 13 depending on resolution of input and required responsiveness
#ifdef  ANGLE_BITS
#define ANGLE_RES			(1<<ANGLE_BITS)
#else
#define ANGLE_RES			(h_dims.x)
#endif//ANGLE_BITS
// how much to multiply by to shift the most significant angle bit to the msb of a short
#define ANGLE_16_BITS		((1<<16)/ANGLE_RES)
//TODO: check if this actually calculates or if the compiler is smart enough
#define ANGLE_SCALE			(2*M_PI_F/ANGLE_RES)
#define ANGLE_FLIP			(ANGLE_RES>>1)
#define DIVERGENCE_THRESH	(1<<(15-3))

union vec2ToArr {
	float2 v;
	float a[2];
};

// [1] In	fF4_src_image: 2 channel masked image of strong x and y gradient (SFLOAT), 
//				angle (SNORM), and gradient magnitude (UFLOAT)
// [2] Out	fc1_dst_image: 1 channel Hough (lines) accumulator (UFLOAT), x-axis 
//				reperesents rotation 0 to 360, y-axis represents signed distance along 
//				edge gradient direction to center of image. This aids in finding 
//				opposing gradient pairs and more evenly distributes Hough space 
//				information density.
//NOTE: this kernel needs to be driven by the OUTPUT dimensions which should 
// have a width that is a multiple of 2, recommended 2048 for 720p, this 
// prevents needing barriers to avoid output write collisions
//NOTE: similar rho values should have similar amounts of work so work groups 
// should be arranged to keep rho constant within them
__kernel void hough_lines(read_only image2d_t fF4_src_image, write_only image2d_t fc1_dst_image)
{
	// determine ouput coordinate for work item
	int2 h_coords = (int2)(get_global_id(0), get_global_id(1));	//TODO: probably need extra logic here to better load balance
	int2 h_dims = get_image_dim(fc1_dst_image);
	float2 i_dims = convert_float2(get_image_dim(fF4_src_image) & -2);

	// convert from normal (polar) to normal (cartesian) form
	union vec2ToArr grad;
	grad.a[1] = sincos(h_coords.x * ANGLE_SCALE, grad.a);
	// account for center offset parameterization choice while we're at it
	float2 normal = fma(grad.v, h_coords.y - HOUGH_HEIGHT/2, i_dims / 2);

	// direction to step along the line is 90 degree to gradient/normal
	float2 step = grad.v.yx * (float2)(-1.0f, 1.0f);

//	if(get_local_linear_id()==0)
//		printf("normal=(%f,%f) h_coords=(%i,%i) step=(%f,%f) h_angle=%f half_height=%i\n", normal.x, normal.y, h_coords.x, h_coords.y, step.x, step.y, h_angle, half_height);
	// convert coords to a signed 16 bit fixed precision angle representation that wraps properly
	short h_angle16 = h_coords.x * ANGLE_16_BITS;
	i_dims -= 1;	// no indexing past the last pixel
	float accum[2] = {0};
	float2 i_coords = normal;
	bool twice = false;
	do
	{
		// check if coords are within image bounds, otherwise it will sample garbage
		while(all((i_coords >= 0) & (i_coords <= i_dims)))
		{
			// read the gradient vector and angle at the current point
			//NOTE: due to discontinuity at +/- 180 degrees, and sampling with bilinear the angle can be wildly off
			// I *THINK* it should end up reading 180 degrees off if that's the case which isn't a problem since
			// the angle is just used for thresholding and we keep both ones that are aligned and anti-aligned
			// but I didn't verify this
			float3 i_grad = read_imagef(fF4_src_image, convert_int(i_coords)).xyz;
			// convert image gradient angle from +/- 1.0f to 16 bit fixed precision angle,
			// take difference between proposed normal, if difference is near zero, it's aligned,
			// if it's near +/- 32768 it's anti-aligned and multiplying by 2 causes it to wrap to near 0 as well
			// this enables proccessing both at the same time
			short angle_divergence = abs((short)((convert_short_rte(i_grad.z * (1<<15)) - h_angle16)<<1));
			if(angle_divergence < DIVERGENCE_THRESH)
			{
				float product = dot(grad.v, i_grad.lo);
				accum[product < 0] += product*product;
			}
			i_coords += step;
		}
		step *= -1;
		i_coords = normal + step;
	} while(!twice++);

	write_imagef(fc1_dst_image, h_coords, (float4)(accum[0]/32, 0, 0, 1));
	write_imagef(fc1_dst_image, (int2)(h_coords.x ^ ANGLE_FLIP, (HOUGH_HEIGHT - 1 - h_coords.y)), (float4)(accum[1]/32, 0, 0, 1));
}