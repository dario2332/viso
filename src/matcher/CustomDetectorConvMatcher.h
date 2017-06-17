#ifndef __CUSTOM_CONV_MATCHER_H__
#define __CUSTOM_CONV_MATCHER_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <emmintrin.h>
#include <algorithm>
#include <vector>
#include <random>
#include <string>
#include "cblas.h"

#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvdef.h>

#include "matrix.h"
#include "filter.h"
#include "ConvMatcher.h"
#include "FeatureExtractor.h"

using namespace cv;

class CustomDetectorConvMatcher : public ConvMatcher {

public:
  CustomDetectorConvMatcher(parameters param, string &graph_path, Ptr<Feature2D> detector) : ConvMatcher(param, graph_path), detector(detector) {} ;
  virtual ~CustomDetectorConvMatcher(); 

//  virtual void bucketFeatures(int32_t max_features,float bucket_width,float bucket_height);

 // virtual void pushBackFetures (string left, string right="");
  virtual void pushBackFeatures (shared_ptr<ImageDescriptor> left, shared_ptr<ImageDescriptor> right);

protected:

  void detect (Mat &img, vector<KeyPoint> &points, int cell_size=10, int num_per_cell = 5);
  void nonMaxSuppression(vector<KeyPoint> &points, Mat &img, vector<Matcher::maximum> &maxima, float min_distance);
  virtual void computeFeatures (uint8_t *I,const int32_t* dims,int32_t* &max1,int32_t &num1,int32_t* &max2,int32_t &num2,uint8_t* &I_du,uint8_t* &I_dv,uint8_t* &I_du_full,uint8_t* &I_dv_full);

private:
  Ptr<Feature2D> detector;
  Mat left_img;
  Mat right_img;
 
};

#endif

