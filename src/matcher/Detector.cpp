#include "Detector.h"

void Detector::nonMaxSuppression(vector<KeyPoint> &points, float min_distance) {

  int max_row = 0;
  int max_col = 0;
  for (auto point : points) {
    if (point.pt.x > max_col) max_col = point.pt.x;
    if (point.pt.y > max_row) max_row = point.pt.y;
  }
  Mat response_matrix = Mat::zeros(max_row + min_distance, max_col + min_distance, CV_32S);

  sort(points.begin(), points.end(), [this](KeyPoint &point1, KeyPoint& point2) {return this->compareFeatures(point1, point2);});
  points.insert(points.begin(), KeyPoint(0, 0, 0));

  for (int i = 0; i < points.size(); ++i) {
    int x = round(points[i].pt.y);
    int y = round(points[i].pt.x);
    if (response_matrix.at<int>(x, y) == 0) 
      response_matrix.at<int>(x, y) = i;
  }

  vector<KeyPoint> filtered_points;
  for (int p_i = 1; p_i < points.size(); ++p_i) {
    int x = round(points[p_i].pt.y);
    int y = round(points[p_i].pt.x);
    int center_index = response_matrix.at<int>(x, y);
    if (center_index == 0) continue;
    if (x == 0 && y == 0) continue;
 
    for (int i_patch = x < min_distance ? 0 : x-min_distance; i_patch < response_matrix.rows && i_patch <= x + min_distance; ++i_patch) {
      for (int j_patch = y < min_distance ? 0 : y-min_distance; j_patch < response_matrix.cols && j_patch <= y + min_distance; ++j_patch) {
        if (i_patch != x || j_patch != y) {
          response_matrix.at<int>(i_patch, j_patch) = 0;
        }
      }
    }
    filtered_points.push_back(points[p_i]);
  }
  points = filtered_points;
}


bool Detector::compareFeatures(const KeyPoint &point1, const KeyPoint &point2) const {
  return point1.response > point2.response;
}

void Harris::detectFeatures(Mat img, vector<KeyPoint> &points) {
  //cout << "Start" << endl;
  Mat out;
  float max = -100000;
  float min = 100000;
  //Mat dst, dst_norm, dst_norm_scaled;
  //dst = Mat::zeros( img.size(), CV_32FC1 );
 
    // Detecting corners
  cornerHarris( img, out, 2, 11, 0.05, BORDER_DEFAULT );
 
  //cv::cornerHarris(img, out, 7, 5, 0.05);
  //normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  //convertScaleAbs( dst_norm, dst_norm_scaled );
  //dst_norm_scaled = dst_norm;
 
  //imshow( "corners_window", out );
  //waitKey(0);
  for (int i = 0; i < out.rows; ++i) {
    for (int j = 0; j < out.cols; ++j) {
      float response = out.at<float>(i, j);
      if (response > 100)
        points.push_back(KeyPoint(j, i, out.at<float>(i, j)));
      //cout << out.at<float>(i, j) << endl;
      
      if (response > max) max = response;
      if (response < min) min = response;
    }
  }

  //cout << max << " " << min << endl;
  //cout << "End" << endl;
}

shared_ptr<Detector> DetectorFactory::constructDetector(int d) {
  if (d == ORB) {
    Ptr<Feature2D> cv_detector = cv::ORB::create(20000, 1.2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 2);
    auto ptr = shared_ptr<Detector>(new OpenCVFeature2dWrapper(cv_detector));
    return ptr;
  }
  if (d == Harris) {
    auto ptr = shared_ptr<Detector>(new ::Harris());
    return ptr;
  }
  return nullptr;
}
