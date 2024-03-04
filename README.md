# Ellipse-Recognition
An OpenCL implementation of fast ellipse recognition (WIP)

Currently includes a Scharr filter based edge detector that simultaneously 
calculates x and y gradients from a single input channel. No performance testing 
has been done yet. Ellipse recognition has not yet been implemented.

# Attribution
Thanks to Sean Barrett for his stb image reading and writing libraries found 
here: https://github.com/nothings/stb . They were a great help with testing my 
code and very easy to use. I was pleasantly surprised when I found that reading 
a single channel from an image automatically converted it to greyscale since 
that was exactly what I needed.