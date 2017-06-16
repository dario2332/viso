
#include "ConvMatcher.h"
#include <cmath>
#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvdef.h>


using namespace std;

ConvMatcher::ConvMatcher(parameters param, string &graph_path) : Matcher(param) {
}

ConvMatcher::~ConvMatcher() {
}

shared_ptr<ImageDescriptor> ConvMatcher::getExtractor(int32_t* m) {
  if (m == m1p1 || m == m1p2) {
    return left_prev_features;
  }
  else if (m == m1c1 || m == m1c2) {
    return left_curr_features;
  }
  else if (m == m2c1 || m == m2c2) {
    return right_curr_features;
  }
  else if (m == m2p1 || m == m2p2) {
    return right_prev_features;
  }
  else {
    cout << "ERROR" << endl;
  }
}

void ConvMatcher::findMatch (int32_t* m1,const int32_t &i1,int32_t* m2,const int32_t &step_size,vector<int32_t> *k2,
                                const int32_t &u_bin_num,const int32_t &v_bin_num,const int32_t &stat_bin,
                                int32_t& min_ind,int32_t stage,bool flow,bool use_prior,double u_,double v_) {
  
  
  shared_ptr<ImageDescriptor> first_image = getExtractor(m1);
  shared_ptr<ImageDescriptor> second_image = getExtractor(m2);


  // init and load image coordinates + feature
  min_ind          = 0;
  double  max_similarity = -1;
  int32_t u1       = *(m1+step_size*i1+0);
  int32_t v1       = *(m1+step_size*i1+1);
  int32_t c        = *(m1+step_size*i1+3);
  
  float u_min,u_max,v_min,v_max;
  
  // restrict search range with prior
  if (use_prior) {
    u_min = u1+ranges[stat_bin].u_min[stage];
    u_max = u1+ranges[stat_bin].u_max[stage];
    v_min = v1+ranges[stat_bin].v_min[stage];
    v_max = v1+ranges[stat_bin].v_max[stage];
    
  // otherwise: use full search space
  } else {
    u_min = u1-param.match_radius;
    u_max = u1+param.match_radius;
    v_min = v1-param.match_radius;
    v_max = v1+param.match_radius;
  }
  
  // if stereo search => constrain to 1d
  if (!flow) {
    v_min = v1-param.match_disp_tolerance;
    v_max = v1+param.match_disp_tolerance;
  }
  

  // bins of interest
  int32_t u_bin_min = min(max((int32_t)floor(u_min/(float)param.match_binsize),0),u_bin_num-1);
  int32_t u_bin_max = min(max((int32_t)floor(u_max/(float)param.match_binsize),0),u_bin_num-1);
  int32_t v_bin_min = min(max((int32_t)floor(v_min/(float)param.match_binsize),0),v_bin_num-1);
  int32_t v_bin_max = min(max((int32_t)floor(v_max/(float)param.match_binsize),0),v_bin_num-1);
  
  float *first = first_image->getFeature(u1, v1);
  // for all bins of interest do
  for (int32_t u_bin=u_bin_min; u_bin<=u_bin_max; u_bin++) {
    for (int32_t v_bin=v_bin_min; v_bin<=v_bin_max; v_bin++) {
      int32_t k2_ind = (c*v_bin_num+v_bin)*u_bin_num+u_bin;
      for (vector<int32_t>::const_iterator i2_it=k2[k2_ind].begin(); i2_it!=k2[k2_ind].end(); i2_it++) {
        int32_t u2   = *(m2+step_size*(*i2_it)+0);
        int32_t v2   = *(m2+step_size*(*i2_it)+1);
        if (u2>=u_min && u2<=u_max && v2>=v_min && v2<=v_max) {
          float *second = second_image->getFeature(u2, v2);
          float similarity = cblas_sdot(left_curr_features->d, first, 1, second, 1);

          //if (u_>=0 && v_>=0) {
          //  double du = (double)u2-u_;
          //  double dv = (double)v2-v_;
          //  double dist = sqrt(du*du+dv*dv);
          //  cost += 4*dist;
          //}
          
          if (similarity > max_similarity) {
            min_ind  = *i2_it;
            max_similarity = similarity;
          }
        }
      }
    }
  }
  //cout << max_similarity << endl; //" " << left_prev_features->data[0] << " " << left_curr_features->data[0] << " " << right_prev_features->data[0] << " " << right_curr_features->data[0] << endl;
}

float ConvMatcher::matchScore(const p_match &match, bool only_left_score) {
  float *left_curr = left_curr_features->getFeature(round(match.u1c), round(match.v1c));
  float *right_curr = right_curr_features->getFeature(round(match.u2c), round(match.v2c));
  float *left_prev = left_prev_features->getFeature(round(match.u1p), round(match.v1p));
  float *right_prev = right_prev_features->getFeature(round(match.u2p), round(match.v2p));

  int u1c = round(match.u1c);
  int v1c = round(match.v1c);
  int u1p = round(match.u1p);
  int v1p = round(match.v1p);


  float score = cblas_sdot(left_curr_features->d, left_prev, 1, left_curr, 1);
  if (only_left_score) return score;
  score *= cblas_sdot(left_curr_features->d, left_curr, 1, right_curr, 1);
  if (score < 0) return 0;
  score *= cblas_sdot(left_curr_features->d, left_prev, 1, right_prev, 1);
  if (score < 0) return 0;
  score *= cblas_sdot(left_curr_features->d, right_curr, 1, right_prev, 1);
  return score;
}

bool ConvMatcher::fixMatch(const float &u1, const float &v1, float &u2, float &v2, shared_ptr<ImageDescriptor> first, shared_ptr<ImageDescriptor> second, int n) {

  int u1_ = round(u1);
  int v1_ = round(v1);
  int u2_ = round(u2);
  int v2_ = round(v2);

  float *desc1 = first->getFeature(u1_, v1_);
  float *desc2 = second->getFeature(u2_, v2_);

  bool changed = false;
  float score = cblas_sdot(first->d, desc1, 1, desc2, 1);
  for (int i = u2_-n; i <= u2_+n; ++i) {
    for (int j = v2_-n; j <= v2_+n; ++j) {
      desc2 = second->getFeature(i, j);
      float s = cblas_sdot(first->d, desc1, 1, desc2, 1);
      if (s > score) {
        score = s;
        //cout << match.u1c << " " << match.v1c << endl;
        u2 = i;
        v2 = j;
        if (u2_ != i || v2_ != j)
          changed = true;
      }
    }
  }
  return changed;
}

void ConvMatcher::fixMatches(vector<p_match> &matches) {
  static int unfixed = 0;
  static int fixed = 0;
    
  //static int counter = 0;
  //cv::namedWindow("point1", cv::WINDOW_NORMAL);
  //cv::namedWindow("point2", cv::WINDOW_NORMAL);
  bool uf = true;
    for (int i = 0; i < matches.size(); ++i) {
      //cout << i << endl;
      //int counter = 0;
      uf = true;
      while (uf) {
        bool u1 = fixMatch(matches[i].u1c, matches[i].v1c, matches[i].u1p, matches[i].v1p, left_curr_features, left_prev_features);
        //cv::Mat img = imread(left_curr_features->image_path, cv::IMREAD_GRAYSCALE);
        //cv::Rect roi(matches[i].u1c-20, matches[i].v1c-20, 40, 40);
        //cv::drawKeypoints(img, vector<cv::KeyPoint> {cv::KeyPoint(matches[i].u1c, matches[i].v1c, 0)}, img);
        //cv::Mat area(img, roi);
        //cv::imshow("point1", area);

        //cv::Mat img2 = imread(left_prev_features->image_path, cv::IMREAD_GRAYSCALE);
        //cv::Rect roi2(matches[i].u1p-20, matches[i].v1p-20, 40, 40);
        //cv::drawKeypoints(img2, vector<cv::KeyPoint> {cv::KeyPoint(matches[i].u1p, matches[i].v1p, 0)}, img2);
        //cv::Mat area2(img2, roi2);
        //cv::imshow("point2", area2);
        ////if (u1) counter++;
        bool u2 = fixMatch(matches[i].u1p, matches[i].v1p, matches[i].u1c, matches[i].v1c, left_prev_features, left_curr_features);
        ////if (u2) counter++;
        uf = u1 || u2;
        //cout << counter << endl;
        //cv::waitKey(0);
      }

      //uf = true;
      //while (uf) {
      //  bool u1 = fixMatch(matches[i].u1c, matches[i].v1c, matches[i].u2c, matches[i].v2c, left_curr_features, right_curr_features);
      //  bool u2 = fixMatch(matches[i].u1p, matches[i].v1p, matches[i].u1c, matches[i].v1c, left_prev_features, left_curr_features);
      //  uf = u1 || u2;
      //}

      //uf = true;
      //while (uf) {
      //  bool u1 = fixMatch(matches[i].u1p, matches[i].v1p, matches[i].u2p, matches[i].v2p, left_prev_features, right_prev_features);
      //  bool u2 = fixMatch(matches[i].u2p, matches[i].v2p, matches[i].u1p, matches[i].v1p, right_prev_features, left_prev_features);
      //  uf = u1 || u2;
      //}

      //fixMatch(matches[i].u1p, matches[i].v1p, matches[i].u1c, matches[i].v1c, left_prev_features, left_curr_features);
      //fixMatch(matches[i].u2c, matches[i].v2c, matches[i].u1c, matches[i].v1c, right_curr_features, left_curr_features);
      //fixMatch(matches[i].u2p, matches[i].v2p, matches[i].u1p, matches[i].v1p, right_prev_features, left_prev_features);

      //uf = fixMatch(matches[i].u1c, matches[i].v1c, matches[i].u1p, matches[i].v1p, left_curr_features, left_prev_features);
      //if (uf) unfixed++;
      //else fixed++;
      //uf = fixMatch(matches[i].u1c, matches[i].v1c, matches[i].u2c, matches[i].v2c, left_curr_features, right_curr_features);
      //if (uf) unfixed++;
      //else fixed++;
      //uf = fixMatch(matches[i].u1p, matches[i].v1p, matches[i].u2p, matches[i].v2p, left_prev_features, right_prev_features);
      //if (uf) unfixed++;
      //else fixed++;

      //uf = fixMatch(matches[i].u1p, matches[i].v1p, matches[i].u1c, matches[i].v1c, left_prev_features, left_curr_features);
      //if (uf) unfixed++;
      //else fixed++;
      //uf = fixMatch(matches[i].u2c, matches[i].v2c, matches[i].u1c, matches[i].v1c, right_curr_features, left_curr_features);
      //if (uf) unfixed++;
      //else fixed++;
      //uf = fixMatch(matches[i].u2p, matches[i].v2p, matches[i].u1p, matches[i].v1p, right_prev_features, left_prev_features);
      //if (uf) unfixed++;
      //else fixed++;
    }
  //cout << "Unfixed: " << fixed*1.0 / (unfixed+fixed)  << endl;
}

void ConvMatcher::sortMatchesByScore(vector<p_match> &matches, bool only_left_score) {
  sort(matches.begin(), matches.end(), [this, only_left_score](p_match &m1, p_match& m2) {return matchScore(m1, only_left_score) > matchScore(m2, only_left_score);});
  //for (int i = 0; i < matches.size(); ++i) {
  //  //cout << matchScore(matches[i]) << " ";
  //  fixMatch(matches[i]);
  //  //matchScore(matches[i]);
  //  //cout << endl;
  //}
  //if (matches.size() > 0)
  //  cout << endl;
}

void ConvMatcher::bucketFeatures(int32_t max_features,float bucket_width,float bucket_height) {

  // find max values
  float u_max = 0;
  float v_max = 0;
  for (vector<p_match>::iterator it = p_matched_2.begin(); it!=p_matched_2.end(); it++) {
    if (it->u1c>u_max) u_max=it->u1c;
    if (it->v1c>v_max) v_max=it->v1c;
  }

  // allocate number of buckets needed
  int32_t bucket_cols = (int32_t)floor(u_max/bucket_width)+1;
  int32_t bucket_rows = (int32_t)floor(v_max/bucket_height)+1;
  vector<p_match> *buckets = new vector<p_match>[bucket_cols*bucket_rows];

  // assign matches to their buckets
  for (vector<p_match>::iterator it=p_matched_2.begin(); it!=p_matched_2.end(); it++) {
    int32_t u = (int32_t)floor(it->u1c/bucket_width);
    int32_t v = (int32_t)floor(it->v1c/bucket_height);
    buckets[v*bucket_cols+u].push_back(*it);
  }

  //srand(time(0));
  //srand(0);
  // refill p_matched from buckets
  p_matched_2.clear();
  for (int32_t i=0; i<bucket_cols*bucket_rows; i++) {
    
    // shuffle bucket indices randomly
    std::random_shuffle(buckets[i].begin(),buckets[i].end());

    //fixMatches(buckets[i]);

    if (param.sort == 1)
      sortMatches(buckets[i]);
    if (param.sort == 2)
      sortMatchesByScore(buckets[i], true);
    if (param.sort == 3)
      sortMatchesByScore(buckets[i], false);

    updateMatches(buckets[i]);
    
    // add up to max_features features from this bucket to p_matched
    int32_t k=0;
    for (vector<p_match>::iterator it=buckets[i].begin(); it!=buckets[i].end(); it++) {
      p_matched_2.push_back(*it);
      k++;
      if (k>=max_features)
        break;
    }
  }

  //cout << "bucketing " << p_matched_2.size() << endl;

  // free buckets
  delete []buckets;
}


//void ConvMatcher::pushBackFetures (string left, string right) {
//  static int counter = 0;
//  swap(left_prev_features, left_curr_features);
//  swap(right_prev_features, right_curr_features);
//
//  left_curr_features->extractFeatures(left);
//  if (right != "")
//    right_curr_features->extractFeatures(right);
//}

void ConvMatcher::pushBackFeatures (shared_ptr<ImageDescriptor> left, shared_ptr<ImageDescriptor> right) {
  swap(left_prev_features, left_curr_features);
  swap(right_prev_features, right_curr_features);

  left_curr_features = left;
  right_curr_features = right;

  //if (right != "")
    //right_curr_features->extractFeatures(right);
}

//void ConvMatcher::setDims(int W, int H, int D) {
//  left_prev_features->initDims(W, H, D);
//  left_curr_features->initDims(W, H, D);
//  right_prev_features->initDims(W, H, D);
//  right_curr_features->initDims(W, H, D);
//}

void ConvMatcher::computeDescriptor (const int32_t &u,const int32_t &v, float *desc_addr) {
  float* features = left_curr_features->getFeature(u, v);
  std::copy (features, features+64, desc_addr);
}
