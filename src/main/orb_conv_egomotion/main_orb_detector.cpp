

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

#include "OrbMatcher.h"
// These are all common classes it's handy to reference with no namespace.
using namespace std;
using namespace cv;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


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
  string graph_path="trained/model.pb";

  ofstream output_file;
  output_file.open(result_file.c_str());
  
  // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  Egomotion::parameters param;
  loadCalibParams(param, dir);
  param.ransac_iters = 10000;
  param.inlier_threshold = 1.0;
  param.reweighting = false;

  //matcher = cv::ORB::create(10000, 1.2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 2);
  Ptr<Feature2D> detector = cv::ORB::create(20000);
  OrbMatcher::Params matcher_params;
  matcher_params.ns_min_distance = 4;
  matcher_params.nonMaxSuppression = true;
  matcher_params.sort = true;
  matcher_params.bucket_size = 2;
  matcher_params.stereo_offset = 1;

  OrbMatcher matcher(graph_path, detector, matcher_params);
  // init visual odometry
  Egomotion viso_main(param);
  
  // current pose (this matrix transforms a point from the current
  // frame's camera coordinates to the first frame's camera coordinates)
  Matrix pose = Matrix::eye(4);

  srand(time(0));

  for (int32_t i=0; i<num_frames; i++) {

    // input file names
    //char base_name[256]; sprintf(base_name,"%06d.png",i);
    char base_name[256]; sprintf(base_name,"%06d",i);
    string left_img_file_name  = dir + "/image_0/" + base_name + ".png";
    string right_img_file_name = dir + "/image_1/" + base_name + ".png";
    
    matcher.pushBack(left_img_file_name, right_img_file_name);

    Matrix final_T;

    //process normal frame
    if (viso_main.process(matcher.getMatches())) {
      final_T = Matrix::inv(viso_main.getMotion());
      //Oprez ovo su inlieri sa obzirom na translaciju nakon rotacije
      cout << viso_main.getNumberOfInliers() << endl;
      auto matches = matcher.getMatches();
      writeMatches(matches, "../matches/orb/" + std::to_string(i));
    }
    else {
      final_T = Matrix::eye(4);
      cout << " ... failed! " << i << endl;
    }

    pose = pose * final_T;
    
    if (i % 25 == 0) {
      cout << dir << ": " << i << "/" << num_frames << endl;
    }
      
    //save pose
    writePose(output_file, pose);
    if (i < num_frames - 1) {
      output_file << endl;
    }

  }
  
  // output
  cout << "Demo complete! Exiting ..." << endl;
  output_file.close();

  // exit
  return 0;
}

