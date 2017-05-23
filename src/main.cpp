/*
Copyright 2012. All rights reserved.
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


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdint.h>

#include "slerp.h"
#include "util.h"
#include "viso_stereo_seperate.h"
#include <png++/png.hpp>
#include <boost/qvm/all.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/surface_matching.hpp>

using namespace cv;
using namespace std;


int main (int argc, char** argv) {

  // we need the path name to 2010_03_09_drive_0019 as input argument
  if (argc<4) {
    cerr << "Usage: ./viso2 path/to/sequence/ num_frames result_file" << endl;
    return 1;
  }


  // sequence directory
  string dir = argv[1];
  int num_frames = atoi(argv[2]);
  string result_file = argv[3];

  ofstream output_file;
  output_file.open(result_file.c_str());
  
  // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  VisualOdometryStereoSeperate::parameters param;
  loadCalibParams(param, dir);
  param.match.refinement = 2;//2;
  param.match.half_resolution = 0;
  param.bucket.max_features = 2000;
  param.match.use_initial_descriptor = false;
  param.match.sort = 0;
  param.match.multi_stage = 1;

  // init visual odometry
  VisualOdometryStereoSeperate viso_main(param);
  
  // current pose (this matrix transforms a point from the current
  // frame's camera coordinates to the first frame's camera coordinates)
  Matrix pose = Matrix::eye(4);

  for (int32_t i=0; i<num_frames; i++) {

    // input file names
    //char base_name[256]; sprintf(base_name,"%06d.png",i);
    char base_name[256]; sprintf(base_name,"%06d.png",i);
    string left_img_file_name  = dir + "/image_0/" + base_name;
    string right_img_file_name = dir + "/image_1/" + base_name;
    
    // catch image read/write errors here
    try {

      // load left and right input image
      png::image< png::gray_pixel > left_img(left_img_file_name);
      png::image< png::gray_pixel > right_img(right_img_file_name);

      // image dimensions
      int32_t width  = left_img.get_width();
      int32_t height = left_img.get_height();


      // convert input images to uint8_t buffer
      uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));

      int32_t k=0;
      for (int32_t v=0; v<height; v++) {
        for (int32_t u=0; u<width; u++) {
          left_img_data[k]  = left_img.get_pixel(u,v);
          right_img_data[k] = right_img.get_pixel(u,v);
          k++;
        }
      }

      int32_t dims[] = {width,height,width};
      
      Matrix final_T;

      //process normal frame
      if (viso_main.process(left_img_data, right_img_data, dims)) {
        final_T = Matrix::inv(viso_main.getMotion());
        auto matches = viso_main.getMatches();
        writeMatches(matches, "../matches/sad/" + std::to_string(i));
      }
      else {
        final_T = Matrix::eye(4);
        cout << " ... failed! " << i << endl;
      }

      pose = pose * final_T;
      
      if (i % 100 == 0) {
        cout << dir << ": " << i << "/" << num_frames << endl;
      }
      
      //save pose
      writePose(output_file, pose);
      if (i < num_frames - 1) {
        output_file << endl;
      }

      // release uint8_t buffers
      free(left_img_data);
      free(right_img_data);

    // catch image read errors here
    } catch (...) {
      cerr << "ERROR: Couldn't read input files!" << endl;
      return 1;
    }
  }
  
  // output
  cout << "Demo complete! Exiting ..." << endl;
  output_file.close();

  // exit
  return 0;
}

