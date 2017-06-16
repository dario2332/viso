
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
#include "FeatureExtractor.h"

class ConvMatcher : public Matcher {

public:
  ConvMatcher(parameters param, string &graph_path);
  virtual ~ConvMatcher(); 

  virtual void bucketFeatures(int32_t max_features,float bucket_width,float bucket_height);

//  virtual void pushBackFetures (string left, string right="");
  virtual void pushBackFeatures (shared_ptr<ImageDescriptor> left, shared_ptr<ImageDescriptor> right);

  void computeDescriptor (const int32_t &u,const int32_t &v, float *desc_addr);

  void setDims(int W, int H, int D=64);

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
  void calculateDotProducts(vector<p_match> &matches);

  shared_ptr<ImageDescriptor> left_prev_features;
  shared_ptr<ImageDescriptor> right_prev_features;
  shared_ptr<ImageDescriptor> left_curr_features;
  shared_ptr<ImageDescriptor> right_curr_features;
 
};

#endif

