#ifndef OFFSETS_LUT_CL
#define OFFSETS_LUT_CL
//TODO: go through files and see which need to be switched to using this
constant const int2 offsets[] = {(int2)(1,0),1,(int2)(0,1),(int2)(-1,1),(int2)(-1,0),-1,(int2)(0,-1),(int2)(1,-1)};

#endif//OFFSETS_LUT_CL