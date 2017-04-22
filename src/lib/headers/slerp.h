#ifndef SLERP_H
#define SLERP_H

#include "matrix.h"
#include <boost/qvm/all.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/surface_matching.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <iostream>

using namespace boost::qvm;
using namespace std;

quat<double> createQuaternion(Matrix R);
Matrix createRotation(quat<double> q);
void printQuaternion(quat<double> q);

Matrix estimateRotationSlerp(Matrix R_1_2, Matrix R_1_3, Matrix R_2_3);

#endif // SLERP_H
