# Ellipse-Recognition
An OpenCL implementation of fast ellipse recognition (WIP)

Currently includes a Scharr filter based edge detector that simultaneously 
calculates x and y gradients from a single input channel. No performance testing 
has been done yet. Ellipse recognition has not yet been implemented.

# Cloning this repo
This repo uses git submodules to pull in the OpenCL headers, so you must use
```git submodule update --init --recursive``` to pull in those before compiling for the
first time. You also need to get the compiled OpenCL binaries for your platform
from somewhere. Official SDK binaries can be found here:
* TODO: add link to binaries
Then you must edit the OPENCL_ROOT variable on line 11 of the Makefile to point 
to the directory. If after compiling one of the applications, it fails to run 
with no output or errors, then it probably needs the DLL to be copied to the 
same directory as the executable. I do not know why this happens as I've had it 
just work and pick up the proper DLL from the Windows system files when compiled
on one computer and not see any instances of OpenCL.DLL when compiled on another
despite using the same files.

# Attribution
Thanks to Sean Barrett for his stb image reading and writing libraries found 
here: https://github.com/nothings/stb . They were a great help with testing my 
code and very easy to use. I was pleasantly surprised when I found that reading 
a single channel from an image automatically converted it to greyscale since 
that was exactly what I needed.

# TODO List
* Add type read/write type mismatch warning for OpenCL kernel compilation by 
parsing raw input for mis-matched read/write calls
* verify artificial vector bithacks actually provide a perf benefit
* verify early exits for various kernels actually provide a perf benefit
* attempt hash-style fast reduce and see if there's a significant perf benefit 
over single threading
* split compilation into a separate program
* add configuration hashing for auto-rebuild support (low priority)
* add a second debug kernel compile list so they aren't all mixed together in
the same list
* make the boilerplate fully data driven so that recompilation of the C portion
only happens for serious changes to how the program launches/outputs data, also 
would get rid of all the "unused variable" warnings
* FIXME: somewhere in the boilerplate I missed a check for the underlying type 
code (u/i) on the creation of the buffer and instead relied on capitalization of 
the type/width code which can also have that info but is a lot easier to get wrong
* arc_segments needs to be partially rewritten to avoid thread divergence
* everything needs to be gone over with a fine tooth comb because I'm a dummy and
forgot the importance of using group shared memory when possible
* add support for re-using arguments for write_only parameters (colored_retrace_starts.cl)
* add support for non-image type kernel arguments (serial_reduce/_lines.cl)