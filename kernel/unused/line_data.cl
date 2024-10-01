#ifndef LINE_DATA_CL
#define LINE_DATA_CL

#define IS_NOT_END		(1 << 0)
#define IS_CW			(1 << 1)
#define IS_BOTH_HANDED	(1 << 2)

struct line_data{
	char2 offset_end;
	uchar flags;	// collection of flags: 0b0000bte	// no bitfield suppprt in OpenCL :(
	// "e" represents that line segment creation thread didn't end processing on this line, inverted so that empty pixels evaluate as ends
	// "t" represents if the line appears to turn cw to the next (or previous if end) line in the chain
	// "b" represents if the line should be counted to both turning directions processing due to ambiguity
	uchar len;
};

struct line_AB_tracking{
	struct line_data data[2];	// previous line's length and, flags are cleared on write
	int2 base_coords[2];
	char curr;					// updated automatically on write, DO NOT MANUALLY SET
};

// used only for reading/writing line_data to an image buffer
union _line_rw{
	struct line_data data;
	uint ui;
};

//len, end flag, base coords, and offset mid must be pre-populated, rest get calculated here before writing
void write_line_data(write_only image2d_t ui1_line_data, struct line_AB_tracking* lines, int2 coords)
{	//TODO: update flags before write, currently just skeleton code
	lines->curr = !lines->curr;	// toggle current write slot
	const char prev = lines->curr;	// alias for code readability

	//avoid writing length 0 lines, only 1st line in chain should ever have length 0 in the 2nd index
	if(lines->data[prev].len == 0)
		return;
	
	write_imageui(ui1_line_data, lines->base_coords[prev], ((union _line_rw)lines->data[prev]).ui);
}
#endif//LINE_DATA_CL