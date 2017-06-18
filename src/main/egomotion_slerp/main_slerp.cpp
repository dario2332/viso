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
#include "Egomotion.h"
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
    cerr << "Usage: ./viso2 path/to/sequence/ num_frames result_file result_file_slerp" << endl;
    return 1;
  }


  // sequence directory
  string dir = argv[1];
  int num_frames = atoi(argv[2]);
  string result_file = argv[3];
  string result_file_slerp = argv[4];

  ofstream output_file;
  ofstream output_file_slerp;
  output_file.open(result_file.c_str());
  output_file_slerp.open(result_file_slerp.c_str());
  
  // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  Egomotion::parameters param;
  loadCalibParams(param, dir);
  param.match.refinement = 2;
  param.match.half_resolution = 0;
  param.bucket.max_features = 2;
  param.match.use_initial_descriptor = true;
  param.match.sort = 1;

  // init visual odometry
  Egomotion viso_main(param);
  Egomotion viso_odd(param);
  Egomotion viso_even(param);

  
  // current pose (this matrix transforms a point from the current
  // frame's camera coordinates to the first frame's camera coordinates)
  Matrix pose = Matrix::eye(4);
  Matrix pose_slerp = Matrix::eye(4);

  Matrix T_0_1;
  bool process_0_1 = false;

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
      
      bool process_0_2=false, process_1_2=false;
      Matrix final_T, T_0_2, T_1_2;

      //process normal frame
      process_1_2 = viso_main.process(left_img_data, right_img_data, dims);

      //process frame for slerp
      if (i % 2 == 0) {
        process_0_2 = viso_even.process(left_img_data, right_img_data, dims);
      }
      else {
        process_0_2 = viso_odd.process(left_img_data, right_img_data, dims);
      }

      //set transformation for normal estimation
      if (process_1_2) {
        T_1_2 = Matrix::inv(viso_main.getMotion());
      }
      else {
        T_1_2 = Matrix::eye(4);
      }

      //slerp
      if (process_0_1 && process_0_2 && process_1_2) {
        T_0_2 = (i % 2 == 0) ? Matrix::inv(viso_even.getMotion()) : Matrix::inv(viso_odd.getMotion());

        Matrix rotation = estimateRotationSlerp(T_0_1.getMat(0, 0, 2, 2), T_0_2.getMat(0, 0, 2, 2), T_1_2.getMat(0, 0, 2, 2));

        if (viso_main.calculateTranslation(Matrix::inv(rotation))) {
          final_T = Matrix::inv(viso_main.getMotion());
        }
        else {
          cout << "slerp failed" << i << endl;
        }
      }
      
      //if slerp failed use normal transformation
      if (final_T.m == 0 && process_1_2) {
        final_T = T_1_2;
      }
      //esle use identity matrix
      else if (final_T.m == 0) {
        final_T = Matrix::eye(4);
        cout << " ... failed! " << i << endl;
      }

      //update both poses
      pose_slerp = pose_slerp * final_T;
      pose = pose * T_1_2;
      
      if (i % 100 == 0) {
        cout << dir << ": " << i << "/" << num_frames << endl;
      }
      
      //save both poses
      writePose(output_file, pose);
      writePose(output_file_slerp, pose_slerp);
      if (i < num_frames - 1) {
        output_file << endl;
        output_file_slerp << endl;
      }
      //update old transformation
      //T_0_1 = T_1_2;
      T_0_1 = final_T;
      process_0_1 = process_1_2;

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
  output_file_slerp.close();

  // exit
  return 0;
}

