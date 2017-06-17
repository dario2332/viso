
#include "CustomDetectorConvMatcher.h"


void CustomDetectorConvMatcher::nonMaxSuppression(vector<KeyPoint> &points, Mat &img, vector<Matcher::maximum> &maxima, float min_distance) {
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

  //cv::drawKeypoints(img, filtered_points, img);
  //cv::imshow("keypoints", img);
  //cv::waitKey(0);
  //vector<Point2f> poi;
  for (auto point : filtered_points) {
    //cout << point.pt.x << " " << point.pt.y << endl;
    //static int counter = 0;
    //cout << "------" << endl;
    //poi.push_back(Point2f(round(point.pt.x), round(point.pt.y)));
    //cv::cornerSubPix(img, poi, cv::Size(1, 1), cv::Size(-1, -1), TermCriteria(TermCriteria::Type::COUNT + TermCriteria::Type::EPS, 100, 0.001));
    maxima.push_back(Matcher::maximum(round(point.pt.x), round(point.pt.y), point.response, 0));
    //cout << poi[0].x << " " << poi[0].y << endl;
    //poi.clear();
    //if (counter == 10)
    //  exit(1);
  }
  //points = filtered_points;
}


void CustomDetectorConvMatcher::detect (Mat &img, vector<KeyPoint> &points, int cell_size, int num_per_cell) {
  int margin = 31;
  cout << "start" << endl;
  for (int i = margin; i+cell_size <= img.cols-margin; i+=cell_size) {
    for (int j = margin; j+cell_size <= img.rows-margin; j+=cell_size) {
      
      cv::Rect roi(max(i-margin, 0), max(j-margin, 0), cell_size+2*margin, cell_size+2*margin);
      Mat area(img, roi);
      vector<KeyPoint> p;
      detector->detect(area, p);
      for (auto point : p) {
        point.pt.x += i-margin;
        point.pt.y += j-margin;
        points.push_back(point);
      }
    }
  }
  cout << "end" << endl;
}

void CustomDetectorConvMatcher::computeFeatures (uint8_t *I,const int32_t* dims,int32_t* &max1,int32_t &num1,int32_t* &max2,int32_t &num2,
                      uint8_t* &I_du,uint8_t* &I_dv,uint8_t* &I_du_full,uint8_t* &I_dv_full) {
  //Matcher::computeFeatures(I, dims, max1, num1, max2, num2, I_du, I_dv, I_du_full, I_dv_full);
  //return;

  int16_t *I_f1;
  int16_t *I_f2;
  
  int32_t dims_matching[3];
  memcpy(dims_matching,dims,3*sizeof(int32_t));
  
  // allocate memory for sobel images and filter images
  if (!param.half_resolution) {
    I_du = (uint8_t*)_mm_malloc(dims[2]*dims[1]*sizeof(uint8_t*),16);
    I_dv = (uint8_t*)_mm_malloc(dims[2]*dims[1]*sizeof(uint8_t*),16);
    I_f1 = (int16_t*)_mm_malloc(dims[2]*dims[1]*sizeof(int16_t),16);
    I_f2 = (int16_t*)_mm_malloc(dims[2]*dims[1]*sizeof(int16_t),16);
    filter::sobel5x5(I,I_du,I_dv,dims[2],dims[1]);
    filter::blob5x5(I,I_f1,dims[2],dims[1]);
    filter::checkerboard5x5(I,I_f2,dims[2],dims[1]);
  } else {
    uint8_t* I_matching = createHalfResolutionImage(I,dims);
    getHalfResolutionDimensions(dims,dims_matching);
    I_du      = (uint8_t*)_mm_malloc(dims_matching[2]*dims_matching[1]*sizeof(uint8_t*),16);
    I_dv      = (uint8_t*)_mm_malloc(dims_matching[2]*dims_matching[1]*sizeof(uint8_t*),16);
    I_f1      = (int16_t*)_mm_malloc(dims_matching[2]*dims_matching[1]*sizeof(int16_t),16);
    I_f2      = (int16_t*)_mm_malloc(dims_matching[2]*dims_matching[1]*sizeof(int16_t),16);
    I_du_full = (uint8_t*)_mm_malloc(dims[2]*dims[1]*sizeof(uint8_t*),16);
    I_dv_full = (uint8_t*)_mm_malloc(dims[2]*dims[1]*sizeof(uint8_t*),16);
    filter::sobel5x5(I_matching,I_du,I_dv,dims_matching[2],dims_matching[1]);
    filter::sobel5x5(I,I_du_full,I_dv_full,dims[2],dims[1]);
    filter::blob5x5(I_matching,I_f1,dims_matching[2],dims_matching[1]);
    filter::checkerboard5x5(I_matching,I_f2,dims_matching[2],dims_matching[1]);
    _mm_free(I_matching);
  }
  

  vector<KeyPoint> curr_points;
  Mat &img = left_img;
  if (I == I2c)
    img = right_img;

  detector->detect(img, curr_points);


  // extract sparse maxima (1st pass) via non-maximum suppression
  vector<Matcher::maximum> maxima1;
  if (param.multi_stage) {
    int32_t nms_n_sparse = param.nms_n*3;
    if (nms_n_sparse>10)
      nms_n_sparse = max(param.nms_n,10);
    nonMaxSuppression(curr_points, img, maxima1, nms_n_sparse);
    computeDescriptors(I_du,I_dv,dims_matching[2],maxima1);
  }
  
  // extract dense maxima (2nd pass) via non-maximum suppression
  vector<Matcher::maximum> maxima2;
  nonMaxSuppression(curr_points, img, maxima2, param.nms_n);
  computeDescriptors(I_du,I_dv,dims_matching[2],maxima2);

  // release filter images
  _mm_free(I_f1);
  _mm_free(I_f2);  
  
  // get number of interest points and init maxima pointer to NULL
  num1 = maxima1.size();
  num2 = maxima2.size();
  //cout << num1 << endl;
  //cout << num2 << endl;
  max1 = 0;
  max2 = 0;
  
  int32_t s = 1;
  if (param.half_resolution)
    s = 2;

  // return sparse maxima as 16-bytes aligned memory
  if (num1!=0) {
    max1 = (int32_t*)_mm_malloc(sizeof(Matcher::maximum)*num1,16);
    int32_t k=0;
    for (vector<Matcher::maximum>::iterator it=maxima1.begin(); it!=maxima1.end(); it++) {
      *(max1+k++) = it->u*s;  *(max1+k++) = it->v*s;  *(max1+k++) = 0;        *(max1+k++) = it->c;
      *(max1+k++) = it->d1;   *(max1+k++) = it->d2;   *(max1+k++) = it->d3;   *(max1+k++) = it->d4;
      *(max1+k++) = it->d5;   *(max1+k++) = it->d6;   *(max1+k++) = it->d7;   *(max1+k++) = it->d8;
    }
  }
  
  // return dense maxima as 16-bytes aligned memory
  if (num2!=0) {
    max2 = (int32_t*)_mm_malloc(sizeof(Matcher::maximum)*num2,16);
    int32_t k=0;
    for (vector<Matcher::maximum>::iterator it=maxima2.begin(); it!=maxima2.end(); it++) {
      *(max2+k++) = it->u*s;  *(max2+k++) = it->v*s;  *(max2+k++) = 0;        *(max2+k++) = it->c;
      *(max2+k++) = it->d1;   *(max2+k++) = it->d2;   *(max2+k++) = it->d3;   *(max2+k++) = it->d4;
      *(max2+k++) = it->d5;   *(max2+k++) = it->d6;   *(max2+k++) = it->d7;   *(max2+k++) = it->d8;
    }
  }
}

//void CustomDetectorConvMatcher::pushBackFetures (string left, string right) {
//  left_img = cv::imread(left, cv::IMREAD_GRAYSCALE);
//  right_img = cv::imread(right, cv::IMREAD_GRAYSCALE);
//  ConvMatcher::pushBackFetures(left, right);
//}

void CustomDetectorConvMatcher::pushBackFeatures (shared_ptr<ImageDescriptor> left, shared_ptr<ImageDescriptor> right) {
  left_img = cv::imread(left->image_path, cv::IMREAD_GRAYSCALE);
  right_img = cv::imread(right->image_path, cv::IMREAD_GRAYSCALE);
  ConvMatcher::pushBackFeatures(left, right);
}

CustomDetectorConvMatcher::~CustomDetectorConvMatcher() { }

void nonMaxSuppression(vector<KeyPoint> &points, Mat &img, float min_distance) {
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


