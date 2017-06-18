
#include "OrbMatcher.h"
#include <cmath>
#include <algorithm>

using namespace std;

OrbMatcher::OrbMatcher(string &graph_path, Ptr<Feature2D> detector, OrbMatcher::Params params) : detector(detector), params(params) {
}

OrbMatcher::~OrbMatcher() {
}

float OrbMatcher::complexMetric(const KeyPoint &point1, const KeyPoint &point2, ConvNetwork *first_extractor, ConvNetwork *second_extractor) {
  int frame_radius = 10;

  int u1 = round(point1.pt.x) - frame_radius;
  u1 = max(0, u1);
  int v1 = round(point1.pt.y) - frame_radius;
  v1 = max(0, v1);
  int u2 = round(point2.pt.x) - frame_radius;
  u2 = max(0, u2);
  int v2 = round(point2.pt.y) - frame_radius;
  v2 = max(0, v2);

  float score = 0;
  for (int i = 0; i <= 2*frame_radius; i+=frame_radius/2) {
    for (int j = 0; j <= 2*frame_radius; j+=frame_radius/2) {
      float *first = first_extractor->getFeature(u1+i, v1+i);
      float *second = second_extractor->getFeature(u2+i, v2+i);
      score += cblas_sdot(first_extractor->D, first, 1, second, 1);
    }
  }
  return score;
}

vector<int> OrbMatcher::oneWayMatching(const vector<KeyPoint> &points1, const vector<KeyPoint> &points2, 
                                       ConvNetwork *first_extractor, ConvNetwork *second_extractor, vector<float> &similarity) {
  int max_distance = params.matching_radius;
  vector<int> p1_to_p2;
  int top_n = 10;

  for (int i = 0; i < points1.size(); ++i) {
    vector<int> top_indexes;
    vector<float> top_sims;
    
    float max_simmilarity = -1;
    int index = -1;
    int u1 = round(points1[i].pt.x);
    int v1 = round(points1[i].pt.y);
    float *first = first_extractor->getFeature(u1, v1);

    for (int j = 0; j < points2.size(); ++j) {
      int u2 = round(points2[j].pt.x);
      int v2 = round(points2[j].pt.y);
      int du = u1-u2, dv = v1-v2;

      if (du*du + dv*dv < max_distance*max_distance) {
        float *second = second_extractor->getFeature(u2, v2);
        float similarity = cblas_sdot(first_extractor->D, first, 1, second, 1);
        if (similarity > max_simmilarity) {
          if (top_indexes.size() < top_n) {
            top_indexes.push_back(j);
            top_sims.push_back(similarity);
            if (top_indexes.size() == top_n) {
              max_simmilarity = *std::min_element(top_sims.begin(), top_sims.end());
            } 
          }
          else {
            int min_index = std::min_element(top_sims.begin(), top_sims.end()) - top_sims.begin();
            top_indexes[min_index] = j;
            top_sims[min_index] = similarity;
            max_simmilarity = *std::min_element(top_sims.begin(), top_sims.end());
          }
        }
      }
    }

    for (int j = 0; j < top_indexes.size(); ++j) {
      top_sims[j] = complexMetric(points1[i], points2[top_indexes[j]], first_extractor, second_extractor);
    }

    int max_index = std::max_element(top_sims.begin(), top_sims.end()) - top_sims.begin();
    if (top_indexes.size() > 0) {
      p1_to_p2.push_back(top_indexes[max_index]);
      similarity.push_back(top_sims[max_index]);
    }
    //cout << top_sims[max_index] << endl;
  }
  return p1_to_p2;
}


void OrbMatcher::nonMaxSuppression(vector<KeyPoint> &points, Mat &img, float min_distance) {
  Mat response_matrix = Mat::zeros(img.rows, img.cols, CV_32S);

  sort(points.begin(), points.end(), [](KeyPoint &point1, KeyPoint& point2) {return point1.response > point2.response;});
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

void OrbMatcher::findDisparity(float &u, float &v, ConvNetwork *left_extractor, ConvNetwork *right_extractor) {
  int disp_max = params.disp_max;

  int u_l = round(u);
  int v_l = round(v);

  float *left = left_extractor->getFeature(u_l, v_l);

  float max_simmilarity = -1;
  int disparity = 0;

  float final_u = u;
  float final_v = v;

  for (int j = v_l -params.stereo_offset; j <= v_l + params.stereo_offset; j++) {
    for (int i = 0; i < disp_max; ++i) {
      if (u_l - i >= 0) {
        float *right = right_extractor->getFeature(u_l-i, j);
        float similarity = cblas_sdot(left_extractor->D, left, 1, right, 1);
        if (similarity > max_simmilarity) {
          max_simmilarity = similarity;
          disparity = i;
          final_u = u - disparity;
          final_v = v + j - v_l;
        }
      }
    }
  }
  u = final_u;
  v = final_v;
}

void OrbMatcher::computeMatches(vector<KeyPoint> &curr_points) {
  if (prev_points.size() == 0 || curr_points.size() == 0) return;

  vector<float> curr_to_prev_sim;
  vector<float> prev_to_curr_sim;
  //cout << "start" << endl;
  vector<int> curr_to_prev = oneWayMatching(curr_points, prev_points, left_curr_features, left_prev_features, curr_to_prev_sim);
  vector<int> prev_to_curr = oneWayMatching(prev_points, curr_points, left_prev_features, left_curr_features, prev_to_curr_sim);
  //cout << "end" << endl;

  matches.clear();

  float treshold = 0.0;
  for (int i = 0; i < curr_to_prev.size(); ++i) {
    //circle closed
    if (prev_to_curr[curr_to_prev[i]] == i && curr_to_prev_sim[i] >= treshold && prev_to_curr_sim[curr_to_prev[i]] >= treshold) {
      Matcher::p_match match;
      //left current
      match.u1c = curr_points[i].pt.x;
      match.v1c = curr_points[i].pt.y;

      //right current
      match.v2c = match.v1c;
      match.u2c = match.u1c;
      findDisparity(match.u2c, match.v2c, left_curr_features, right_curr_features);

      //left prev
      match.u1p = prev_points[curr_to_prev[i]].pt.x;
      match.v1p = prev_points[curr_to_prev[i]].pt.y;

      //right prev
      match.v2p = match.v1p;
      match.u2p = match.u1p;
      findDisparity(match.u2p, match.v2p, left_prev_features, right_prev_features);

      matches.push_back(match);
    }
  }

  bucketFeatures(params.bucket_size, params.bucket_width, params.bucket_height);
}

float OrbMatcher::matchScore(const Matcher::p_match &match) {
  float *left_curr = left_curr_features->getFeature(round(match.u1c), round(match.v1c));
  float *right_curr = right_curr_features->getFeature(round(match.u2c), round(match.v2c));
  float *left_prev = left_prev_features->getFeature(round(match.u1p), round(match.v1p));
  float *right_prev = right_prev_features->getFeature(round(match.u2p), round(match.v2p));

  float score = 1;
  score *= cblas_sdot(left_curr_features->D, left_curr, 1, right_curr, 1);
  if (score < 0) return 0;
  score *= cblas_sdot(left_curr_features->D, left_prev, 1, left_curr, 1);
  if (score < 0) return 0;
  score *= cblas_sdot(left_curr_features->D, left_prev, 1, right_prev, 1);
  if (score < 0) return 0;
  score *= cblas_sdot(left_curr_features->D, right_curr, 1, right_prev, 1);
  return score;
}

void OrbMatcher::sortMatchesByScore(vector<Matcher::p_match> &matches) {
  sort(matches.begin(), matches.end(), [this](Matcher::p_match &m1, Matcher::p_match& m2) {return matchScore(m1) > matchScore(m2);});
  //for (int i = 0; i < matches.size(); ++i) {
  //  cout << matchScore(matches[i]) << " ";
  //}
  //if (matches.size() > 0)
  //  cout << endl;
}


void OrbMatcher::bucketFeatures(int32_t max_features,float bucket_width,float bucket_height) {

  cout << "Num matches before bucketing: " << matches.size() << endl;
  // find max values
  float u_max = 0;
  float v_max = 0;
  for (vector<Matcher::p_match>::iterator it = matches.begin(); it!=matches.end(); it++) {
    if (it->u1c>u_max) u_max=it->u1c;
    if (it->v1c>v_max) v_max=it->v1c;
  }

  // allocate number of buckets needed
  int32_t bucket_cols = (int32_t)floor(u_max/bucket_width)+1;
  int32_t bucket_rows = (int32_t)floor(v_max/bucket_height)+1;
  vector<Matcher::p_match> *buckets = new vector<Matcher::p_match>[bucket_cols*bucket_rows];

  // assign matches to their buckets
  for (vector<Matcher::p_match>::iterator it=matches.begin(); it!=matches.end(); it++) {
    int32_t u = (int32_t)floor(it->u1c/bucket_width);
    int32_t v = (int32_t)floor(it->v1c/bucket_height);
    buckets[v*bucket_cols+u].push_back(*it);
  }
  
  //srand(time(0));
  //srand(0);
  // refill p_matched from buckets
  //cout << "bucketing " << matches.size() << endl;
  matches.clear();
  for (int32_t i=0; i<bucket_cols*bucket_rows; i++) {
    
    // shuffle bucket indices randomly
    std::random_shuffle(buckets[i].begin(),buckets[i].end());
    
    // add up to max_features features from this bucket to p_matched
    if (params.sort)
      sortMatchesByScore(buckets[i]);
    int32_t k=0;
    for (vector<Matcher::p_match>::iterator it=buckets[i].begin(); it!=buckets[i].end(); it++) {
      int d;
      //it->v2c = it->v1c;
      //it->u2c = it->u1c;
      //findDisparity(it->u2c, it->v2c, left_curr_features, right_curr_features);
      //if (d < 0) continue;
      //it->u2c = it->u1c - d;

      //it->v2p = it->v1p;
      //it->u2p = it->u1p;
      //findDisparity(it->u2p, it->v2p, left_prev_features, right_prev_features);
      //if (d < 0) continue;
      //it->u2p = it->u1p - d;
      matches.push_back(*it);

      k++;
      if (k>=max_features)
        break;
    }
  }


  cout << "Num matches after bucketing: " << matches.size() << endl;
  // free buckets
  delete []buckets;
}


void OrbMatcher::pushBack (string left, string right) {
  static int counter = 0;
  swap(left_prev_features, left_curr_features);
  swap(right_prev_features, right_curr_features);

  Mat left_img = cv::imread(left, cv::IMREAD_GRAYSCALE);
  if (left_prev_features->D == 0) {
    setDims(left_img.cols, left_img.rows, 64);
  }

  vector<KeyPoint> curr_points;
  detector->detect(left_img, curr_points);
  
  if (params.nonMaxSuppression)
    nonMaxSuppression(curr_points, left_img, params.ns_min_distance);

  cout << "Num features: " << curr_points.size() << endl;
  left_curr_features->extractFeatures(left);
  if (right != "")
    right_curr_features->extractFeatures(right);

  //cv::drawKeypoints(left_img, curr_points, left_img);
  //cv::imshow("keypoints", left_img);
  //cv::waitKey(0);
  computeMatches(curr_points);
  prev_points = curr_points;
}

void OrbMatcher::setDims(int W, int H, int D) {
  left_prev_features->initDims(W, H, D);
  left_curr_features->initDims(W, H, D);
  right_prev_features->initDims(W, H, D);
  right_curr_features->initDims(W, H, D);
}

void OrbMatcher::computeDescriptor (const int32_t &u,const int32_t &v, float *desc_addr) {
  float* features = left_curr_features->getFeature(u, v);
  std::copy (features, features+64, desc_addr);
}
