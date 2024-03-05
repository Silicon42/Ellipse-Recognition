constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
constant sampler_t lin_samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
//TODO: check if it's cheaper to manually filter between the adjacent pixels since it uses half the pixels

// [0] IN	src_image: 4 channel image of gradient magnitude (UFLOAT), angle (SNORM)(UNUSED), and x and y normalized gradient (SNORM)
// [1] OUT	dst_image: 1 channel image mask (UINT) with result 1 = (local max && low threshold met) and 2 = (local max && high threshold met)
__kernel void canny(read_only image2d_t src_image, write_only image2d_t dst_image, float2 thresh)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 grad = read_imagef(src_image, sampler, coords);
	int2 mask = thresh < grad.x;
	if (!mask.lo)	// guard to hopefully let some workgroups exit early
		return;
	float2 along;
	along.lo = read_imagef(src_image, lin_samp, convert_float2(coords) - grad.hi).x;
	along.hi = read_imagef(src_image, lin_samp, convert_float2(coords) + grad.hi).x;
	if(any(along > grad.x))	// non-max suppression
		return;
	write_imageui(dst_image, coords, 1 - mask.hi);
}