//TODO: go through files and see which need to be switched to using this
constant const int2 offsets[] = {
	(int2)( 1, 0),	//[0]	>
			  1,	//[1]
	(int2)( 0, 1),	//[2]	v
	(int2)(-1, 1),	//[3]
	(int2)(-1, 0),	//[4]	<
			 -1,	//[5]
	(int2)( 0,-1),	//[6]	^
	(int2)( 1,-1)	//[7]
};
constant const char2 offsets_c[] = {(char2)(1,0),1,(char2)(0,1),(char2)(-1,1),(char2)(-1,0),-1,(char2)(0,-1),(char2)(1,-1)};
