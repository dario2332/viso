#ifndef LIBVISO_HELPER
#define LIBVISO_HELPER 

void keypointsToMaxima(vector<KeyPoint> &keypoints, vector<Matcher::maximum> &maxima) {
  for (auto point : keypoints) {
    maxima.push_back(Matcher::maximum(round(point.pt.x), round(point.pt.y), point.response, 0));
  }
}



#endif /* ifndef LIBVISO_HELPER */


