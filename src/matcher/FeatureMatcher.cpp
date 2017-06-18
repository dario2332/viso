
#include "FeatureMatcher.h"
#include "libviso_helper.h"


void FeatureMatcher::computeFeatures (uint8_t *I,const int32_t* dims,int32_t* &max1,int32_t &num1,int32_t* &max2,int32_t &num2,
                      uint8_t* &I_du,uint8_t* &I_dv,uint8_t* &I_du_full,uint8_t* &I_dv_full) {

  if (detector == nullptr) {
    Matcher::computeFeatures(I, dims, max1, num1, max2, num2, I_du, I_dv, I_du_full, I_dv_full);
    return;
  }

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

  Mat img_temp(dims[1], dims[2], CV_8U, I);
  Rect roi(0, 0, dims[0], dims[1]);
  Mat img(img_temp, roi);

  detector->detectFeatures(img, curr_points);

  //cv::drawKeypoints(img, curr_points, img);
  //cv::imshow("keypoints", img);
  //cv::waitKey(0);

  // extract sparse maxima (1st pass) via non-maximum suppression
  vector<Matcher::maximum> maxima1;
  if (param.multi_stage) {
    auto curr_points_copy = curr_points;
    int32_t nms_n_sparse = param.nms_n*3;
    if (nms_n_sparse>10)
      nms_n_sparse = max(param.nms_n,10);
    detector->nonMaxSuppression(curr_points_copy, nms_n_sparse);
    keypointsToMaxima(curr_points_copy, maxima1);
    computeDescriptors(I_du,I_dv,dims_matching[2],maxima1);
  }
  
  // extract dense maxima (2nd pass) via non-maximum suppression
  vector<Matcher::maximum> maxima2;
  detector->nonMaxSuppression(curr_points, param.nms_n);
  keypointsToMaxima(curr_points, maxima2);
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


FeatureMatcher::~FeatureMatcher() { }


