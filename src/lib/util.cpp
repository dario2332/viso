#include "util.h"

using namespace std;
// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
// Assumed R = Rz * Ry * Rx
cv::Vec3d rotationMatrixToEulerAngles(cv::Mat R)
{
     
    double sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    double x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3d(x, y, z);
}

// Assumed R = Rx * Ry * Rz
cv::Vec3d rotationMatrixToEulerAngles(Matrix R)
{
    double sy = sqrt(R.val[0][0] * R.val[0][0] +  R.val[0][1] * R.val[0][1] );
 
    bool singular = sy < 1e-6; // If
 
    double x, y, z;
    if (!singular)
    {
        x = atan2(-R.val[1][2] , R.val[2][2]);
        y = atan2(R.val[0][2], sy);
        z = atan2(-R.val[0][1], R.val[0][0]);
    }
    else
    {
        x = atan2(R.val[2][1], R.val[1][1]);
        y = atan2(R.val[0][2], sy);
        z = 0;
        std::cout << "Singular..." << std::endl;
    }
    return cv::Vec3d(x, y, z);
}

void loadCalibParams(VisualOdometryStereo::parameters &param, std::string dir) {
  std::string s;
  std::ifstream file;
  file.open((dir + "/calib.txt").c_str());
  file >> s;
  double d[12];
  for (int i = 0; i < 12; ++i) {
    file >> d[i];
  }

  param.calib.f = d[0];
  param.calib.cu = d[2];
  param.calib.cv = d[6];
  file >> s;
  for (int i = 0; i < 12; ++i) {
    file >> d[i];
  }
  param.base = - d[3] / d[0];
}

void writePose(ofstream &output_file, Matrix &pose) {
  double data[16];
  pose.getData(data);
  for (int j = 0; j < 12; ++j) {
    output_file << data[j] << " ";
  }
}


void writeMatches (std::vector<Matcher::p_match> &matches, std::string output_file) {
  ofstream out(output_file);
  for (auto match : matches) {
    out << match.u1c << " " << match.v1c << " " << match.u1p << " " << match.v1p << " ";
    out << match.u2c << " " << match.v2c << " " << match.u2p << " " << match.v2p << endl;
  }
}
