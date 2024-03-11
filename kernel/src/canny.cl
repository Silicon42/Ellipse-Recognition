const sampler_t lin_samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
//TODO: check if it's cheaper to manually filter between the adjacent pixels since it would access half the pixels
#define THRESH 0.5

// [0] IN	src_image: 4 channel image of x and y gradient (SFLOAT), angle (SNORM),
//				and gradient magnitude (UFLOAT)
// [1] OUT	dst_image: 4 channel masked src_image angle and magnitude transferred
//				directly and x and y gradient normalized for pixels 
//				meeting the gradient threshold and passing non-max suppression
//NOTE: Doesn't implement the hysteresis portion since that is inherently a very 
// serial operation, blurring and gradient computation is assumed to be already applied
__kernel void canny(read_only image2d_t src_image, write_only image2d_t dst_image)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	float4 grad = read_imagef(src_image, coords);

	if (THRESH > grad.w)	// guard to hopefully let some workgroups exit early
		return;
	
	float2 along;
	float2 n_grad = grad.lo / (float2)grad.w;
	along.lo = read_imagef(src_image, lin_samp, convert_float2(coords) - n_grad).w;
	along.hi = read_imagef(src_image, lin_samp, convert_float2(coords) + n_grad).w;
	if(any(along > grad.w))	// non-max suppression
		return;
	
	grad.lo /= grad.w;
	write_imagef(dst_image, coords, grad);
}