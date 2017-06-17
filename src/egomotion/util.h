#ifndef UTIL_VISO_H
#define UTIL_VISO_H

#include "matrix.h"
#include "matcher.h"
#include "viso_stereo.h"
#include <iostream>
#include <fstream>


#include <opencv2/opencv.hpp>


using namespace std;

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
// Assumed R = Rz * Ry * Rx
cv::Vec3d rotationMatrixToEulerAngles(cv::Mat R);


// Assumed R = Rx * Ry * Rz
cv::Vec3d rotationMatrixToEulerAngles(Matrix R);

void loadCalibParams(VisualOdometryStereo::parameters &param, std::string dir);

void writePose(std::ofstream &output_file, Matrix &pose);

void writeMatches(std::vector<Matcher::p_match> &matches, std::string output_file);

#endif // UTIL_VISO_H
