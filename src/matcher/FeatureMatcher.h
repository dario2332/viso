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

#include "matcher.h"
#include "matrix.h"
#include "filter.h"
#include "ConvNetwork.h"
#include "Detector.h"

using namespace cv;

class FeatureMatcher : public Matcher {

public:
  FeatureMatcher(parameters param, shared_ptr<Detector> detector) : Matcher(param), detector(detector) {} ;
  virtual ~FeatureMatcher(); 

protected:

  virtual void computeFeatures (uint8_t *I,const int32_t* dims,int32_t* &max1,int32_t &num1,int32_t* &max2,int32_t &num2,uint8_t* &I_du,uint8_t* &I_dv,uint8_t* &I_du_full,uint8_t* &I_dv_full);

private:
  //Ptr<Feature2D> detector;
  shared_ptr<Detector> detector;
 
};

#endif

