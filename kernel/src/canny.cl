const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
const sampler_t lin_samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
//TODO: check if it's cheaper to manually filter between the adjacent pixels since it would access half the pixels
#define THRESH 0.5

// [0] IN	src_image: 4 channel image of gradient magnitude (UFLOAT), angle (SNORM), and x and y normalized gradient (SNORM)
// [1] OUT	dst_image: 2 channel masked image with result 1 = (local max && low threshold met) and 2 = (local max && high threshold met)
//NOTE: Doesn't implement the hysteresis portion since that is inherently a very serial operation
__kernel void canny(read_only image2d_t src_image, write_only image2d_t dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 grad = read_imagef(src_image, sampler, coords);

	if (THRESH > grad.x)	// guard to hopefully let some workgroups exit early
		return;
	float2 along;
	along.lo = read_imagef(src_image, lin_samp, convert_float2(coords) - grad.hi).x;
	along.hi = read_imagef(src_image, lin_samp, convert_float2(coords) + grad.hi).x;
	if(any(along > grad.x))	// non-max suppression
		return;
	write_imagef(dst_image, coords, (float4)(grad.lo, 0.5f, 1.0f));
}