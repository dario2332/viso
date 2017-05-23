
#include "ConvMatcher.h"
#include <cmath>

using namespace std;

ConvMatcher::ConvMatcher(parameters param, string &graph_path) : Matcher(param) {
  left_prev_features = new FeatureExtractor(graph_path);
  right_prev_features = new FeatureExtractor(graph_path);
  left_curr_features = new FeatureExtractor(graph_path);
  right_curr_features = new FeatureExtractor(graph_path);
}

ConvMatcher::~ConvMatcher() {
  delete left_prev_features;
  delete right_prev_features;
  delete left_curr_features;
  delete right_curr_features;
}

FeatureExtractor* ConvMatcher::getExtractor(int32_t* m) {
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
  
  
  FeatureExtractor *first_image = getExtractor(m1);
  FeatureExtractor *second_image = getExtractor(m2);


  // init and load image coordinates + feature
  min_ind          = 0;
  double  max_similarity = -1;
  int32_t u1       = *(m1+step_size*i1+0);
  int32_t v1       = *(m1+step_size*i1+1);
  int32_t c        = *(m1+step_size*i1+3);
  __m128i xmm1     = _mm_load_si128((__m128i*)(m1+step_size*i1+4));
  __m128i xmm2     = _mm_load_si128((__m128i*)(m1+step_size*i1+8));
  
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
          float similarity = cblas_sdot(left_curr_features->D, first, 1, second, 1);

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

void ConvMatcher::calculateDotProducts(vector<p_match> &matches) {
  static int counter = 0;
  static int counter_false = 0;
  vector<p_match> filtered_matches;
  for (int i = 0; i < matches.size(); ++i) {
    float *left_curr = left_curr_features->getFeature(matches[i].u1c, matches[i].v1c);
    float *right_curr = right_curr_features->getFeature(matches[i].u2c, matches[i].v2c);
    float a = cblas_sdot(left_curr_features->D, left_curr, 1, right_curr, 1);
    //cout << matches[i].u1c << " " << matches[i].v1c << " " << matches[i].u2c << " " << matches[i].v2c << " " << a << endl;
    int start_counter = counter;
    if (a > 0.8) {
      float *left_prev = left_prev_features->getFeature(matches[i].u1p, matches[i].v1p);
      a = cblas_sdot(left_curr_features->D, left_prev, 1, left_curr, 1);
      if (a > 0.8) {
        float *right_prev = right_prev_features->getFeature(matches[i].u2p, matches[i].v2p);
        a = cblas_sdot(left_curr_features->D, left_prev, 1, right_prev, 1);
        if (a > 0.8) {
          float *right_curr = right_curr_features->getFeature(matches[i].u2c, matches[i].v2c);
          a = cblas_sdot(left_curr_features->D, right_curr, 1, right_prev, 1);
          if (a > 0.8) 
            filtered_matches.push_back(matches[i]);
        }
      }
    }
    if (start_counter == counter) counter_false++;
    //if (matches[i].v1c == matches[i].v2c) {
    //  cout << a << " ";
    //  float *left_prev = left_prev_features->getFeature(matches[i].u1p, matches[i].v1p);
    //  float a_prev = cblas_sdot(left_curr_features->D, left_prev, 1, left, 1);
    //  cout << a_prev << endl;
    //}
  }
  matches = filtered_matches;
  //cout << counter << " " << counter_false << endl;
}

float ConvMatcher::matchScore(const p_match &match) {
  float *left_curr = left_curr_features->getFeature(round(match.u1c), round(match.v1c));
  float *right_curr = right_curr_features->getFeature(round(match.u2c), round(match.v2c));
  float *left_prev = left_prev_features->getFeature(round(match.u1p), round(match.v1p));
  float *right_prev = right_prev_features->getFeature(round(match.u2p), round(match.v2p));

  float score = 0;
  score += cblas_sdot(left_curr_features->D, left_curr, 1, right_curr, 1);
  score += cblas_sdot(left_curr_features->D, left_prev, 1, left_curr, 1);
  score += cblas_sdot(left_curr_features->D, left_prev, 1, right_prev, 1);
  score += cblas_sdot(left_curr_features->D, right_curr, 1, right_prev, 1);
  return score;
}

void ConvMatcher::sortMatchesByScore(vector<p_match> &matches) {
  sort(matches.begin(), matches.end(), [this](p_match &m1, p_match& m2) {return matchScore(m1) > matchScore(m2);});
  //for (int i = 0; i < matches.size(); ++i) {
  //  cout << matchScore(matches[i]) << " ";
  //}
  //if (matches.size() > 0)
  //  cout << endl;
}

//void ConvMatcher::bucketFeatures(int32_t max_features,float bucket_width,float bucket_height) {
//
//  // find max values
//  float u_max = 0;
//  float v_max = 0;
//  for (vector<p_match>::iterator it = p_matched_2.begin(); it!=p_matched_2.end(); it++) {
//    if (it->u1c>u_max) u_max=it->u1c;
//    if (it->v1c>v_max) v_max=it->v1c;
//  }
//
//  // allocate number of buckets needed
//  int32_t bucket_cols = (int32_t)floor(u_max/bucket_width)+1;
//  int32_t bucket_rows = (int32_t)floor(v_max/bucket_height)+1;
//  vector<p_match> *buckets = new vector<p_match>[bucket_cols*bucket_rows];
//
//  // assign matches to their buckets
//  for (vector<p_match>::iterator it=p_matched_2.begin(); it!=p_matched_2.end(); it++) {
//    int32_t u = (int32_t)floor(it->u1c/bucket_width);
//    int32_t v = (int32_t)floor(it->v1c/bucket_height);
//    buckets[v*bucket_cols+u].push_back(*it);
//  }
//  
//  //srand(time(0));
//  srand(0);
//  // refill p_matched from buckets
//  p_matched_2.clear();
//  for (int32_t i=0; i<bucket_cols*bucket_rows; i++) {
//    
//    // shuffle bucket indices randomly
//    std::random_shuffle(buckets[i].begin(),buckets[i].end());
//    //calculateDotProducts(buckets[i]);
//    if (param.sort == 1)
//      sortMatches(buckets[i]);
//    if (param.sort == 2)
//      sortMatchesByScore(buckets[i]);
//    //updateMatches(buckets[i]);
//    
//    // add up to max_features features from this bucket to p_matched
//    int32_t k=0;
//    for (vector<p_match>::iterator it=buckets[i].begin(); it!=buckets[i].end(); it++) {
//      p_matched_2.push_back(*it);
//      k++;
//      if (k>=max_features)
//        break;
//    }
//  }
//
//  //cout << "bucketing " << p_matched_2.size() << endl;
//
//  // free buckets
//  delete []buckets;
//}

//cblas_sdot(L.D, &L.elements[start_L], 1, &R.elements[start_R], 1);


void ConvMatcher::pushBackFetures (string left, string right) {
  static int counter = 0;
  swap(left_prev_features, left_curr_features);
  swap(right_prev_features, right_curr_features);

  left_curr_features->extractFeatures(left);
  if (right != "")
    right_curr_features->extractFeatures(right);
}


void ConvMatcher::setDims(int W, int H, int D) {
  left_prev_features->initDims(W, H, D);
  left_curr_features->initDims(W, H, D);
  right_prev_features->initDims(W, H, D);
  right_curr_features->initDims(W, H, D);
}

void ConvMatcher::computeDescriptor (const int32_t &u,const int32_t &v, float *desc_addr) {
  float* features = left_curr_features->getFeature(u, v);
  std::copy (features, features+64, desc_addr);
}
