#include "slerp.h"
using namespace std;

quat<double> createQuaternion(Matrix R) {
  cv::ppf_match_3d::Pose3D q;
  double d[9];
  R.getData(d);
  q.updatePose(d, d);
  quat<double> result;
  for (int i = 0; i < 4; ++i) {
    result.a[i] = q.q[i];
  }
  //boost::qvm::normalize(result);
  return result;
}

Matrix createRotation(quat<double> q) {
  cv::ppf_match_3d::Pose3D p;
  p.updatePoseQuat(q.a, q.a);
  Matrix R(3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R.setVal(p.pose[i*4+j], i, j);
    }
  }
  //p.printPose();
  return R;
}

void printQuaternion(quat<double> q) {
  for (int i = 0; i < 4; ++i) {
    cout << q.a[i] << " ";
  }
  cout << endl;
}

//normalize
Matrix estimateRotationSlerp(Matrix R_1_2, Matrix R_1_3, Matrix R_2_3) {
  quat<double> q_1_2 = createQuaternion(R_1_2);
  quat<double> q_1_3 = createQuaternion(R_1_3);
  quat<double> q_2_3 = createQuaternion(R_2_3);
  quat<double> q_ = boost::qvm::inverse(q_1_2) * q_1_3;
  boost::qvm::normalize(q_);
  boost::qvm::normalize(q_2_3);
  quat<double> result = slerp(q_, q_2_3, 0.5);
  //handling NaN
  if (result.a[0] != result.a[0]) {
    cout << "NAAAAAAN" << endl;
    return R_2_3;
  }
  boost::qvm::normalize(result);
  return createRotation(result);
}


