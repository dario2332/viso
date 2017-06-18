#include "Detector.h"

void Detector::nonMaxSuppression(vector<KeyPoint> &points, float min_distance) {

  int max_row = 0;
  int max_col = 0;
  for (auto point : points) {
    if (point.pt.x > max_col) max_col = point.pt.x;
    if (point.pt.y > max_row) max_row = point.pt.y;
  }
  Mat response_matrix = Mat::zeros(max_row, max_col, CV_32S);

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


shared_ptr<Detector> DetectorFactory::constructDetector(int d) {
  if (d == ORB) {
    Ptr<Feature2D> cv_detector = cv::ORB::create(20000, 1.2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 1);
    auto ptr = shared_ptr<Detector>(new OpenCVFeature2dWrapper(cv_detector));
    return ptr;
  }
  return nullptr;
}
