#ifndef DETECTOR_H
#define DETECTOR_H 

#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvdef.h>

#include <vector>
#include <memory>

using namespace std;
using namespace cv;

class Detector {
public:
  virtual void detectFeatures(Mat img, vector<KeyPoint> &points) = 0;
  virtual void nonMaxSuppression(vector<KeyPoint> &curr_points, float min_distance);
  virtual ~Detector () {};

protected:
  //return true if point1 is better feature than point2
  virtual bool compareFeatures(const KeyPoint &point1, const KeyPoint &point2) const;
};

class OpenCVFeature2dWrapper : public Detector {
public:
  OpenCVFeature2dWrapper(Ptr<Feature2D> detector) : detector(detector) {};
  virtual void detectFeatures(Mat img, vector<KeyPoint> &points) {
    detector->detect(img, points);
  };

protected:
  Ptr<Feature2D> detector;
};

class Harris : public Detector {
  virtual void detectFeatures(Mat img, vector<KeyPoint> &points);
};

class DetectorFactory {

public:
  enum {
    ORB,
    Harris
  };
  static shared_ptr<Detector> constructDetector(int d);
};
#endif /* ifndef DETECTOR_H */
