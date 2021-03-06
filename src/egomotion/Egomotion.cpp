/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libviso2.
Authors: Andreas Geiger

libviso2 is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libviso2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libviso2; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#include "Egomotion.h"

using namespace std;

Egomotion::Egomotion(parameters param, Matcher *matcher) : param(param), VisualOdometryStereo(param) {
  if (matcher != nullptr)
    setMatcher(matcher);
}

Egomotion::~Egomotion() {
}

void Egomotion::setMatcher(Matcher* new_matcher) {
  delete matcher;
  matcher = new_matcher;
  matcher->setIntrinsics(param.calib.f,param.calib.cu,param.calib.cv,param.base);
}

cv::Vec3d Egomotion::estimateRotation(vector<Matcher::p_match> &p_matched, Egomotion::parameters param) {
  cv::Mat E, R, t, mask;
  vector<cv::Point2d> points1, points2;
  for (int i = 0; i < p_matched.size(); ++i) {
    points1.push_back(cv::Point2d(p_matched[i].u1p, p_matched[i].v1p));
    points2.push_back(cv::Point2d(p_matched[i].u1c, p_matched[i].v1c));
  }
  
  cv::Point2d pp = cv::Point2d(param.calib.cu, param.calib.cv);

  E = cv::findEssentialMat(points1, points2, param.calib.f, pp,  cv::RANSAC, 0.9999, 1.0, mask);

  int inliers = 0, outliers = 0;
  
  //vector<Matcher::p_match> p_matched_copy;
  recoverPose(E, points1, points2, R, t, param.calib.f, pp, mask);
  for (auto i = mask.begin<uchar>(); i != mask.end<uchar>(); ++i) {
    if ((int)*i == 0) outliers++;
    else {
      inliers++;
      //cout << i-mask.begin<uchar>() << endl;
      //p_matched_copy.push_back(p_matched[i-mask.begin<uchar>()]);
    }
  }
  //p_matched = p_matched_copy;
  cout << inliers << " R/" <<  outliers << endl;
  return rotationMatrixToEulerAngles(R);
}

bool Egomotion::calculateTranslation(Matrix R) {
  setRotation(R);
  param.estimate_rotation = false;
  bool result = updateMotion();
  param.estimate_rotation = true;
  return result;
}

void Egomotion::setRotation(Matrix R) {
  R_ = R;
}

void Egomotion::setTranformation(Matrix T) {
  Tr_delta = T;
}

vector<double> Egomotion::estimateMotion (vector<Matcher::p_match> p_matched) {
  
  // get number of matches
  int32_t N  = p_matched.size();
  if (N<6)
    return vector<double>();

  // return value
  bool success = true;
  
  cv::Vec3d angles;
  //estimate Rotation only with opencv
  if (param.estimate_rotation) {
    angles = estimateRotation(p_matched, param);
  }
  else {
    angles = rotationMatrixToEulerAngles(R_);
  }

  // compute minimum distance for RANSAC samples
  double width=0,height=0;
  for (vector<Matcher::p_match>::iterator it=p_matched.begin(); it!=p_matched.end(); it++) {
    if (it->u1c>width)  width  = it->u1c;
    if (it->v1c>height) height = it->v1c;
  }
  double min_dist = min(width,height)/3.0;
  
  // get number of matches
  N  = p_matched.size();
  if (N<6)
    return vector<double>();

  // allocate dynamic memory
  X          = new double[N];
  Y          = new double[N];
  Z          = new double[N];
  J          = new double[4*N*6];
  p_predict  = new double[4*N];
  p_observe  = new double[4*N];
  p_residual = new double[4*N];

  // project matches of previous image into 3d
  for (int32_t i=0; i<N; i++) {
    double d = max(p_matched[i].u1p - p_matched[i].u2p,0.0001f);
    X[i] = (p_matched[i].u1p-param.calib.cu)*param.base/d;
    Y[i] = (p_matched[i].v1p-param.calib.cv)*param.base/d;
    Z[i] = param.calib.f*param.base/d;
  }

  // loop variables
  vector<double> tr_delta;
  vector<double> tr_delta_curr;
  tr_delta_curr.resize(6, 0);
  
  // clear parameter vector
  inliers.clear();


  for (int32_t i=0; i<3; i++)
    tr_delta_curr[i] = angles[i];
  tr_delta = tr_delta_curr;


  
  if (param.estimate_translation) {
  
  // initial RANSAC estimate
  for (int32_t k=0;k<param.ransac_iters;k++) {

    // draw random sample set
    vector<int32_t> active = getRandomSample(N,3);

    // clear parameter vector
    for (int32_t i=3; i<6; i++)
      tr_delta_curr[i] = 0;


    // minimize reprojection errors
    VisualOdometryStereo::result result = UPDATED;
    int32_t iter=0;
    while (result==UPDATED) {
      result = updateParameters(p_matched,active,tr_delta_curr,1,1e-6);
      if (iter++ > 20 || result==CONVERGED)
        break;
    }

    // overwrite best parameters if we have more inliers
    if (result!=FAILED) {
      vector<int32_t> inliers_curr = getInlier(p_matched,tr_delta_curr);
      if (inliers_curr.size()>inliers.size()) {
        inliers = inliers_curr;
        tr_delta = tr_delta_curr;
      }
    }
  }
  
  
  // final optimization (refinement)
  //cout << "SIZE:" << inliers.size() << endl;
  if (inliers.size()>=6) {
    int32_t iter=0;
    VisualOdometryStereo::result result = UPDATED;
    while (result==UPDATED) {     
      result = updateParameters(p_matched,inliers,tr_delta,1,1e-8);
      if (iter++ > 100 || result==CONVERGED) {
        //cout << "CONVERGED" << endl;
        break;
      }
    }

    if (result != CONVERGED)
      cout << "FAILED" << endl;
    // not converged
    if (result!=CONVERGED)
      success = false;

  // not enough inliers
  } else {
    success = false;
  }
  }


  // release dynamic memory
  delete[] X;
  delete[] Y;
  delete[] Z;
  delete[] J;
  delete[] p_predict;
  delete[] p_observe;
  delete[] p_residual;
  
  // parameter estimate succeeded?
  if (success) return tr_delta;
  else         return vector<double>();
}


Egomotion::result Egomotion::updateParameters(vector<Matcher::p_match> &p_matched,vector<int32_t> &active,vector<double> &tr,double step_size,double eps) {
  
  // we need at least 3 observations
  if (active.size()<3)
    return FAILED;
  
  // extract observations and compute predictions
  computeObservations(p_matched,active);
  computeResidualsAndJacobian(tr,active);

  // init
  Matrix A(3,3);
  Matrix B(3,1);

  // fill matrices A and B
  for (int32_t m=0; m<3; m++) {
    for (int32_t n=0; n<3; n++) {
      double a = 0;
      for (int32_t i=0; i<4*(int32_t)active.size(); i++) {
        a += J[i*3+m]*J[i*3+n];
      }
      A.val[m][n] = a;
    }
    double b = 0;
    for (int32_t i=0; i<4*(int32_t)active.size(); i++) {
      b += J[i*3+m]*(p_residual[i]);
    }
    B.val[m][0] = b;
  }

  // perform elimination
  if (B.solve(A)) {
    bool converged = true;
    for (int32_t m=3; m<6; m++) {
      tr[m] += step_size*B.val[m-3][0];
      if (fabs(B.val[m-3][0])>eps)
        converged = false;
    }
    if (converged)
      return CONVERGED;
    else
      return UPDATED;
  } else {
    return FAILED;
  }
}

void Egomotion::computeResidualsAndJacobian(vector<double> &tr,vector<int32_t> &active) {

  // extract motion parameters
  double rx = tr[0]; double ry = tr[1]; double rz = tr[2];
  double tx = tr[3]; double ty = tr[4]; double tz = tr[5];

  // precompute sine/cosine
  double sx = sin(rx); double cx = cos(rx); double sy = sin(ry);
  double cy = cos(ry); double sz = sin(rz); double cz = cos(rz);

  // compute rotation matrix and derivatives
  double r00    = +cy*cz;          double r01    = -cy*sz;          double r02    = +sy;
  double r10    = +sx*sy*cz+cx*sz; double r11    = -sx*sy*sz+cx*cz; double r12    = -sx*cy;
  double r20    = -cx*sy*cz+sx*sz; double r21    = +cx*sy*sz+sx*cz; double r22    = +cx*cy;

  // loop variables
  double X1p,Y1p,Z1p;
  double X1c,Y1c,Z1c,X2c;
  double X1cd,Y1cd,Z1cd;

  // for all observations do
  for (int32_t i=0; i<(int32_t)active.size(); i++) {

    // get 3d point in previous coordinate system
    X1p = X[active[i]];
    Y1p = Y[active[i]];
    Z1p = Z[active[i]];

    // compute 3d point in current left coordinate system
    X1c = r00*X1p+r01*Y1p+r02*Z1p+tx;
    Y1c = r10*X1p+r11*Y1p+r12*Z1p+ty;
    Z1c = r20*X1p+r21*Y1p+r22*Z1p+tz;
    
    // weighting
    double weight = 1.0;
    if (param.reweighting)
      weight = 1.0/(fabs(p_observe[4*i+0]-param.calib.cu)/fabs(param.calib.cu) + 0.05);
    
    // compute 3d point in current right coordinate system
    X2c = X1c-param.base;

    // for all paramters do
    for (int32_t j=0; j<3; j++) {

      // derivatives of 3d pt. in curr. left coordinates wrt. param j
      switch (j) {
        case 0: X1cd = 1; Y1cd = 0; Z1cd = 0; break;
        case 1: X1cd = 0; Y1cd = 1; Z1cd = 0; break;
        case 2: X1cd = 0; Y1cd = 0; Z1cd = 1; break;
      }

      // set jacobian entries (project via K)
      J[(4*i+0)*3+j] = weight*param.calib.f*(X1cd*Z1c-X1c*Z1cd)/(Z1c*Z1c); // left u'
      J[(4*i+1)*3+j] = weight*param.calib.f*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c); // left v'
      J[(4*i+2)*3+j] = weight*param.calib.f*(X1cd*Z1c-X2c*Z1cd)/(Z1c*Z1c); // right u'
      J[(4*i+3)*3+j] = weight*param.calib.f*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c); // right v'
    }

    // set prediction (project via K)
    p_predict[4*i+0] = param.calib.f*X1c/Z1c+param.calib.cu; // left u
    p_predict[4*i+1] = param.calib.f*Y1c/Z1c+param.calib.cv; // left v
    p_predict[4*i+2] = param.calib.f*X2c/Z1c+param.calib.cu; // right u
    p_predict[4*i+3] = param.calib.f*Y1c/Z1c+param.calib.cv; // right v
    
    // set residuals
    p_residual[4*i+0] = weight*(p_observe[4*i+0]-p_predict[4*i+0]);
    p_residual[4*i+1] = weight*(p_observe[4*i+1]-p_predict[4*i+1]);
    p_residual[4*i+2] = weight*(p_observe[4*i+2]-p_predict[4*i+2]);
    p_residual[4*i+3] = weight*(p_observe[4*i+3]-p_predict[4*i+3]);
  }
}

