*********************************************************
C++ codes are re-implementations of stereo matching algorithms

They are visual studio projects with opencv library 3.0.0 configurations

Inputs are stereo image pairs, outputs are 3D xyz points in saved .csv file and correspondent disparity maps

*********************************************************
Matlab codes are for stereo motion estimation and map building

Inputs of stereo odometry can be set in configfile.m

The .csv above and estimated motions can be used as inputs to the PointCloud_cv.m to build 3D maps
**********************************************************
python code is only a script in Blender, which is used to build the cubes with position and possibility informations from getoccupancy.m 