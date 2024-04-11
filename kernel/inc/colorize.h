// used to convert a single angle channel to a colorized version for better semantic viewing
// alternatively used to provide a unique color to individual "angle" items
// returns a color based on the angle on a color wheel as 3 0.0f to 1.0f values
// input angle is in half revolutions and should be in range +/- 1.0 for best accuracy
float3 colorize(float angle);