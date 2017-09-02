
#ifndef __CONV_MATCHER_H__
#define __CONV_MATCHER_H__

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

#include "matrix.h"
#include "matcher.h"
#include "ConvNetwork.h"
#include "FeatureMatcher.h"

class ConvMatcher : public FeatureMatcher {

public:
  ConvMatcher(parameters param, shared_ptr<Detector> detector = nullptr);
  virtual ~ConvMatcher(); 

  virtual void bucketFeatures(int32_t max_features,float bucket_width,float bucket_height);

  virtual void pushBackFeatures (shared_ptr<ImageDescriptor> left, shared_ptr<ImageDescriptor> right);

  void computeDescriptor (const int32_t &u,const int32_t &v, float *desc_addr);

protected:
  virtual void findMatch (int32_t* m1,const int32_t &i1,int32_t* m2,const int32_t &step_size,
                         std::vector<int32_t> *k2,const int32_t &u_bin_num,const int32_t &v_bin_num,const int32_t &stat_bin,
                         int32_t& min_ind,int32_t stage,bool flow,bool use_prior,double u_=-1,double v_=-1);

private:
  void fixMatches(vector<p_match> &matches);
  bool fixMatch(const float &u1, const float &v1, float &u2, float &v2, shared_ptr<ImageDescriptor> first, shared_ptr<ImageDescriptor> second, int n=1);

  void sortMatchesByScore(vector<p_match> &matches, bool only_left_score=false);
  float matchScore(const p_match &match, bool only_left_score=false);

  shared_ptr<ImageDescriptor> getExtractor(int32_t* m);
  shared_ptr<ImageDescriptor> getExtractor(const uint8_t* m);

  shared_ptr<ImageDescriptor> left_prev_features;
  shared_ptr<ImageDescriptor> right_prev_features;
  shared_ptr<ImageDescriptor> left_curr_features;
  shared_ptr<ImageDescriptor> right_curr_features;

  virtual bool parabolicFitting(const uint8_t* I1_du,const uint8_t* I1_dv,const int32_t* dims1,
                        const uint8_t* I2_du,const uint8_t* I2_dv,const int32_t* dims2,
                        const float &u1,const float &v1,
                        float       &u2,float       &v2,
                        Matrix At,Matrix AtA,
                        uint8_t* desc_buffer);
  virtual void relocateMinimum(const uint8_t* I1_du,const uint8_t* I1_dv,const int32_t* dims1,
                       const uint8_t* I2_du,const uint8_t* I2_dv,const int32_t* dims2,
                       const float &u1,const float &v1,
                       float       &u2,float       &v2,
                       uint8_t* desc_buffer);

};

#endif

