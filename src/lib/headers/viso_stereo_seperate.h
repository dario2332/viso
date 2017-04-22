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

#ifndef VISO_STEREO_SEPERATE_H
#define VISO_STEREO_SEPERATE_H

#include "viso_stereo.h"
#include "util.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

class VisualOdometryStereoSeperate : public VisualOdometryStereo {

public:

  // stereo-specific parameters (mandatory: base)
  struct parameters : public VisualOdometryStereo::parameters {
    bool estimate_translation;
    bool estimate_rotation;
    parameters () {
      estimate_translation = true;
      estimate_rotation = true;
    }
  };

  // constructor, takes as inpute a parameter structure
  VisualOdometryStereoSeperate (parameters param);
  
  // deconstructor
  ~VisualOdometryStereoSeperate ();
  
  void setRotation(Matrix R);
  void setTranformation(Matrix T);

  bool calculateTranslation(Matrix R);
  using VisualOdometryStereo::process;


protected:

  cv::Vec3d estimateRotation(std::vector<Matcher::p_match> &p_matched, VisualOdometry::parameters param);
  virtual std::vector<double>  estimateMotion (std::vector<Matcher::p_match> p_matched);
  virtual result               updateParameters(std::vector<Matcher::p_match> &p_matched,std::vector<int32_t> &active,std::vector<double> &tr,double step_size,double eps);
  virtual void                 computeResidualsAndJacobian(std::vector<double> &tr,std::vector<int32_t> &active);

  Matrix R_;
  parameters param;
};

#endif // VISO_STEREO_SEPERATE_H

