
#ifndef SAMPLERS_CL
#define SAMPLERS_CL

//const sampler_t edge_clamp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
//NOTE: NVidia's implementation of CLK_ADDRESS_CLAMP is out of spec and behaves as CLK_ADDRESS_CLAMP_TO_EDGE instead :(
#define clamped (CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST)

#endif//SAMPLERS_CL