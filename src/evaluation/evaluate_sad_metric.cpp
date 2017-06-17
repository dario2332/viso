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
#include "viso_stereo_seperate.h"
#include <png++/png.hpp>
#include <boost/qvm/all.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/surface_matching.hpp>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using namespace std;
using namespace cv;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

float SADCost(const vector<int32_t> &d1, const vector<int32_t> &d2) {
  __m128i xmm1     = _mm_load_si128((__m128i*)(&d1[0]));
  __m128i xmm2     = _mm_load_si128((__m128i*)(&d1[0]+4));
  __m128i xmm3 = _mm_load_si128((__m128i*)(&d2[0]));
  __m128i xmm4 = _mm_load_si128((__m128i*)(&d2[0]+4));                    
  xmm3 = _mm_sad_epu8 (xmm1,xmm3);
  xmm4 = _mm_sad_epu8 (xmm2,xmm4);
  xmm4 = _mm_add_epi16(xmm3,xmm4);

  float cost = (double)(_mm_extract_epi16(xmm4,0)+_mm_extract_epi16(xmm4,4));
  return cost;
}

void calculateROCPoints(const vector<int> &first, const vector<int> &second, const vector<float> &scores, 
                        const vector<int> &true_labels, const vector<float> &tresholds, vector<float> &imf, vector<float> &cmf) {

  vector<int> tp(tresholds.size(), 0);
  auto tn = tp;
  auto fn = tp;
  auto fp = tp;

  for (int i = 0; i < scores.size(); ++i) {
    bool correct_match = false;
    if (true_labels[first[i]] == true_labels[second[i]]) correct_match = true;
    for (int j = 0; j < tresholds.size(); ++j) {
      if (scores[i] <= tresholds[j] && correct_match) tp[j]++;
      if (scores[i] <= tresholds[j] && !correct_match) fp[j]++;
      if (scores[i] > tresholds[j] && correct_match) fn[j]++;
      if (scores[i] > tresholds[j] && !correct_match) tn[j]++;
    }
  }
  for (int i = 0; i < tresholds.size(); ++i) {
    cmf.push_back(float(tp[i])/(tp[i]+fn[i]));
    imf.push_back(float(fp[i])/(tn[i]+fp[i]));
  }
//  for (int i = 0; i < imf.size(); ++i) {
//    cout <<"TP: " << tp[i] << " FP: " << fp[i] << " TN: " << tn[i] << " FN: " << fn[i] << endl;
//  }
}

int main (int argc, char** argv) {

  vector<float> tresholds;// = {50, 100, 200, 300, 400, 500, 600, 700, 800, 900};
  for (int i = 40; i < 800; ++i) {
    tresholds.push_back(i);
  }
  
  // we need the path name to 2010_03_09_drive_0019 as input argument
  if (argc<4) {
    cerr << "Usage: ./viso2 path/to/sequence/ num_frames result_file" << endl;
    return 1;
  }

  string dir = argv[1];
  int num_frames = atoi(argv[2]);
  string result_file = argv[3];

  ifstream output_file;
  output_file.open(result_file.c_str());

  vector<int> true_labels;
  while (!output_file.eof()) {
    int index;
    output_file >> index;
    true_labels.push_back(index);
    output_file >> index;
  }

  
  // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  VisualOdometryStereoSeperate::parameters param;
  loadCalibParams(param, dir);
  param.match.refinement = 2;
  param.match.half_resolution = 0;
  param.bucket.max_features = 2;
  param.match.use_initial_descriptor = true;
  param.match.sort = 2;
  param.ransac_iters = 1000;

  param.match.nms_n = 3;
  param.match.nms_tau = 50;

  Matcher *matcher  = new Matcher(param.match);

  vector<vector<int32_t> > descriptors;

  for (int32_t i=0; i<num_frames; i++) {

    // input file names
    //char base_name[256]; sprintf(base_name,"%06d.png",i);
    char base_name[256]; sprintf(base_name,"%04d",i);
    string img_file_name  = dir + "/patches" + base_name + ".png";
    
    // catch image read/write errors here
    try {

      png::image< png::gray_pixel > img(img_file_name);

      // image dimensions
      int32_t width  = img.get_width();
      int32_t height = img.get_height();

      // convert input images to uint8_t buffer
      uint8_t* img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));

      int32_t k=0;
      for (int32_t v=0; v<height; v++) {
        for (int32_t u=0; u<width; u++) {
          img_data[k]  = img.get_pixel(u,v);
          k++;
        }
      }

      int32_t dims[] = {width,height,width};
      
      matcher->pushBack(img_data, dims, false);

      for (int v = 0; v < 16; v++) {
        for (int u = 0; u < 16; u++) {
          vector<int32_t> d(8);
          matcher->computeDescriptor(u*width/16 + width/16/2, v*height/16 + height/16/2, (uint8_t*) &d[0]);
          descriptors.push_back(d);
        }
      }

      //if (i % 25 == 0) 
       // cout << i << endl;
      // release uint8_t buffers
      free(img_data);

    // catch image read errors here
    } catch (...) {
      cerr << "ERROR: Couldn't read input files!" << endl;
      return 1;
    }
  }

  //cout << "Size: " << descriptors.size() << endl;

  vector<int> first, second;
  vector<float> scores;

  srand(0);
  int pos_examples = 0;

  for (int i = 0; i < descriptors.size(); ++i) {
    for (int j = i+1; j < descriptors.size(); ++j) {
      if (true_labels[i] != true_labels[j]) break;
      first.push_back(i);
      second.push_back(j);
      //output_file.write(reinterpret_cast<const char *>(&i), sizeof(i));
      //output_file.write(reinterpret_cast<const char *>(&j), sizeof(j));
      float result = SADCost(descriptors[i], descriptors[j]);
      scores.push_back(result);
      //output_file.write(reinterpret_cast<const char *>(&result), sizeof(result));
      pos_examples++;
    }
  //  cout << i << endl;
  }
  int first_elem = 0, second_elem = 0;
  while (pos_examples > 0) {
    first_elem = rand() % descriptors.size();
    second_elem = rand() % descriptors.size();
    if (true_labels[first_elem] != true_labels[second_elem]) {
      first.push_back(first_elem);
      second.push_back(second_elem);

      float result = SADCost(descriptors[first_elem], descriptors[second_elem]);
      scores.push_back(result);
      pos_examples--;
    }
  }
  
  vector<float> imf, cmf;
  calculateROCPoints(first, second, scores, true_labels, tresholds, imf, cmf);
  for (int i = 0; i < imf.size(); ++i) {
    cout << imf[i] << " " << cmf[i] << endl;
  }
  output_file.close();

  // exit
  return 0;
}

