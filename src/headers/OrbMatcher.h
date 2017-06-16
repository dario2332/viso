
#ifndef __ORB_MATCHER_H__
#define __ORB_MATCHER_H__

#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvdef.h>

#include "cblas.h"

#include "matrix.h"
#include "matcher.h"
#include "FeatureExtractor.h"

using namespace cv;

class OrbMatcher {

public:

  struct Params {
    bool sort;
    bool nonMaxSuppression;
    int ns_min_distance;
    int bucket_size;
    int bucket_width;
    int bucket_height;
    int matching_radius;
    int disp_max;
    int stereo_offset;
    
    Params() {
      sort = true;
      nonMaxSuppression = false;
      ns_min_distance = 5;
      bucket_size = 2;
      bucket_width = 50;
      bucket_height = 50;
      matching_radius = 150;
      disp_max = 230;
      stereo_offset = 1;
    }
  };

  OrbMatcher(string &graph_path, Ptr<Feature2D> detector, Params params);
  virtual ~OrbMatcher(); 

//  virtual void bucketFeatures(int32_t max_features,float bucket_width,float bucket_height);

  void pushBack (string left, string right="");

  void computeDescriptor (const int32_t &u,const int32_t &v, float *desc_addr);

  void setDims(int W, int H, int D=64);

  std::vector<Matcher::p_match> getMatches() { return matches; }


private:

  Params params;
  Ptr<Feature2D> detector;
  vector<Matcher::p_match> matches;
  vector<KeyPoint> prev_points;

  FeatureExtractor *left_prev_features;
  FeatureExtractor *right_prev_features;
  FeatureExtractor *left_curr_features;
  FeatureExtractor *right_curr_features;
 
  vector<int> oneWayMatching(const vector<KeyPoint> &points1, const vector<KeyPoint> &points2,
                             FeatureExtractor *first_extractor, FeatureExtractor *second_extractor, vector<float> &similarity);

  void computeMatches(vector<KeyPoint> &curr_points);

  void findDisparity(float &u, float &v, FeatureExtractor *left_extractor, FeatureExtractor *right_extractor);

  void bucketFeatures(int32_t max_features,float bucket_width,float bucket_height);

  float matchScore(const Matcher::p_match &match);
  void sortMatchesByScore(vector<Matcher::p_match> &matches);

  void nonMaxSuppression(vector<KeyPoint> &points, Mat &img, float min_distance = 3);

  float complexMetric(const KeyPoint &point1, const KeyPoint &point2, FeatureExtractor *first_extractor, FeatureExtractor *second_extractor);
};

#endif

