
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

ConvMatcher::ConvMatcher(parameters param, shared_ptr<Detector> detector) : FeatureMatcher(param, detector) {
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

shared_ptr<ImageDescriptor> ConvMatcher::getExtractor(const uint8_t* m) {
  if (m == I1p_du || m == I1p_dv) {
    return left_prev_features;
  }
  else if (m == I1c_du || m == I1c_dv) {
    return left_curr_features;
  }
  else if (m == I2c_du || m == I2c_dv) {
    return right_curr_features;
  }
  else if (m == I2p_du || m == I2p_dv) {
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
      //while (uf) {
        fixMatch(matches[i].u1c, matches[i].v1c, matches[i].u1p, matches[i].v1p, left_curr_features, left_prev_features);
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
        //bool u2 = fixMatch(matches[i].u1p, matches[i].v1p, matches[i].u1c, matches[i].v1c, left_prev_features, left_curr_features);
        ////if (u2) counter++;
        //uf = u1 || u2;
        //cout << counter << endl;
        //cv::waitKey(0);
      //}

      //uf = true;
      //while (uf) {
        fixMatch(matches[i].u1c, matches[i].v1c, matches[i].u2c, matches[i].v2c, left_curr_features, right_curr_features);
      //  bool u2 = fixMatch(matches[i].u1p, matches[i].v1p, matches[i].u1c, matches[i].v1c, left_prev_features, left_curr_features);
      //  uf = u1 || u2;
      //}

      //uf = true;
      //while (uf) {
        fixMatch(matches[i].u1p, matches[i].v1p, matches[i].u2p, matches[i].v2p, left_prev_features, right_prev_features);
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


void ConvMatcher::pushBackFeatures (shared_ptr<ImageDescriptor> left, shared_ptr<ImageDescriptor> right) {
  swap(left_prev_features, left_curr_features);
  swap(right_prev_features, right_curr_features);

  left_curr_features = left;
  right_curr_features = right;

  //if (right != "")
    //right_curr_features->extractFeatures(right);
}


void ConvMatcher::computeDescriptor (const int32_t &u,const int32_t &v, float *desc_addr) {
  float* features = left_curr_features->getFeature(u, v);
  std::copy (features, features+64, desc_addr);
}

bool ConvMatcher::parabolicFitting(const uint8_t* I1_du,const uint8_t* I1_dv,const int32_t* dims1,
                        const uint8_t* I2_du,const uint8_t* I2_dv,const int32_t* dims2,
                        const float &u1,const float &v1,
                        float       &u2,float       &v2,
                        Matrix At,Matrix AtA,
                        uint8_t* desc_buffer) {


  // check if parabolic fitting is feasible (descriptors are within margin)
  if (u2-3<margin || u2+3>dims2[0]-1-margin || v2-3<margin || v2+3>dims2[1]-1-margin)
    return false;
  
  auto first_extractor = getExtractor(I1_du);
  auto second_extractor = getExtractor(I2_du);
  // compute reference descriptor
  
  float *first = first_extractor->getFeature(round(u1), round(v1));
//  __m128i xmm1,xmm2;
//  computeSmallDescriptor(I1_du,I1_dv,dims1[2],(int32_t)u1,(int32_t)v1,desc_buffer);
//  xmm1 = _mm_load_si128((__m128i*)(desc_buffer));
  
  // compute cost matrix
  int32_t cost[49];
  for (int32_t dv=0; dv<7; dv++) {
    for (int32_t du=0; du<7; du++) {
      float *second = second_extractor->getFeature(round(u2+du-3), round(v2+dv-3));
      //computeSmallDescriptor(I2_du,I2_dv,dims2[2],(int32_t)u2+du-3,(int32_t)v2+dv-3,desc_buffer);
      //xmm2 = _mm_load_si128((__m128i*)(desc_buffer));
      //xmm2 = _mm_sad_epu8(xmm1,xmm2);
      cost[dv*7+du] = 40 -10*cblas_sdot(left_curr_features->d, first, 1, second, 1); //_mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4);
    }
  }
  
  // compute minimum
  int32_t min_ind  = 0;
  int32_t min_cost = cost[0];
  for (int32_t i=1; i<49; i++) {
    if (cost[i]<min_cost) {
      min_ind   = i;
      min_cost  = cost[i];
    }
  }
  
  // get indices
  int32_t du = min_ind%7;
  int32_t dv = min_ind/7;
  
  // if minimum is at borders => remove this match
  if (du==0 || du==6 || dv==0 || dv==6)
    return false;
  
  // solve least squares system
  Matrix c(9,1);
  for (int32_t i=-1; i<=+1; i++) {
    for (int32_t j=-1; j<=+1; j++) {
      int32_t cost_curr = cost[(dv+i)*7+(du+j)];
      // if (i!=0 && j!=0 && cost_curr<=min_cost+150)
        // return false;
      c.val[(i+1)*3+(j+1)][0] = cost_curr;
    }
  }
  Matrix b = At*c;
  if (!b.solve(AtA))
    return false;
  
  // extract relative coordinates
  float divisor = (b.val[2][0]*b.val[2][0]-4.0*b.val[0][0]*b.val[1][0]);
  if (fabs(divisor)<1e-8 || fabs(b.val[2][0])<1e-8)
    return false;
  float ddv = (2.0*b.val[0][0]*b.val[4][0]-b.val[2][0]*b.val[3][0])/divisor;
  float ddu = -(b.val[4][0]+2.0*b.val[1][0]*ddv)/b.val[2][0];
  if (fabs(ddu)>=1.0 || fabs(ddv)>=1.0)
    return false;
  
  // update target
  u2 += (float)du-3.0+ddu;
  v2 += (float)dv-3.0+ddv;
  
  // return true on success
  //cout << "true" << endl;
  return true;
}
void ConvMatcher::relocateMinimum(const uint8_t* I1_du,const uint8_t* I1_dv,const int32_t* dims1,
                       const uint8_t* I2_du,const uint8_t* I2_dv,const int32_t* dims2,
                       const float &u1,const float &v1,
                       float       &u2,float       &v2,
                       uint8_t* desc_buffer) { 

  // check if parabolic fitting is feasible (descriptors are within margin)
  if (u2-2<margin || u2+2>dims2[0]-1-margin || v2-2<margin || v2+2>dims2[1]-1-margin)
    return;
  
  auto first_extractor = getExtractor(I1_du);
  auto second_extractor = getExtractor(I2_du);
  // compute reference descriptor
  
  float *first = first_extractor->getFeature(round(u1), round(v1));
  // compute reference descriptor
  //__m128i xmm1,xmm2;
  //computeSmallDescriptor(I1_du,I1_dv,dims1[2],(int32_t)u1,(int32_t)v1,desc_buffer);
  //xmm1 = _mm_load_si128((__m128i*)(desc_buffer));
  
  // compute cost matrix
  int32_t cost[25];
  for (int32_t dv=0; dv<5; dv++) {
    for (int32_t du=0; du<5; du++) {
      float *second = second_extractor->getFeature(round(u2+du-2), round(v2+dv-2));
      //computeSmallDescriptor(I2_du,I2_dv,dims2[2],(int32_t)u2+du-2,(int32_t)v2+dv-2,desc_buffer);
      //xmm2 = _mm_load_si128((__m128i*)(desc_buffer));
      //xmm2 = _mm_sad_epu8(xmm1,xmm2);
      cost[dv*5+du] = 40-10*cblas_sdot(left_curr_features->d, first, 1, second, 1);//_mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4);
    }
  }
  
  // compute minimum
  int32_t min_ind  = 0;
  int32_t min_cost = cost[0];
  for (int32_t i=1; i<25; i++) {
    if (cost[i]<min_cost) {
      min_ind   = i;
      min_cost  = cost[i];
    }
  }
  
  // update target
  u2 += (float)(min_ind%5)-2.0;
  v2 += (float)(min_ind/5)-2.0;

}

