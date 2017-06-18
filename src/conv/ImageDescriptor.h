#ifndef IMG_DESC_H
#define IMG_DESC_H

#include <string>
#include <vector>

using namespace std;

class ImageDescriptor {
  
public:
  ImageDescriptor (int w, int h, int d, string &image_path) : w(w), h(h), d(d), image_path(image_path), data(vector<float> (w*h*d)) {};
  virtual ~ImageDescriptor () {};
  //u-W ,  v-H
  float* getFeature(int u, int v) {
    return &data.at(w * d * v + d * u);
  } 

  int w, d, h;
  vector<float> data;
  string image_path;
};


#endif //IMG_DESC_H
