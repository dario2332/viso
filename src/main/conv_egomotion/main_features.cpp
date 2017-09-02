#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdint.h>

#include "slerp.h"
#include "Egomotion.h"
#include "ConvMatcher.h"

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

#include <thread>
#include <mutex>
#include <queue>
#include <chrono>
// These are all common classes it's handy to reference with no namespace.
using namespace std;
using namespace cv;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


std::mutex m;
std::condition_variable c_v;
queue<shared_ptr<ImageDescriptor> > left_descriptors;
queue<shared_ptr<ImageDescriptor> > right_descriptors;

void loadImages(shared_ptr<ConvNetwork> extractor, string dir, int num_frames) {

  for (int32_t i=0; i<num_frames; i++) {

    char base_name[256]; sprintf(base_name,"%06d",i);
    string left_img_file_name  = dir + "/image_0/" + base_name + ".png";
    string right_img_file_name = dir + "/image_1/" + base_name + ".png";
    shared_ptr<ImageDescriptor> left, right;
    extractor->extractFeatures(left_img_file_name, right_img_file_name, left, right);

    std::unique_lock<std::mutex> lk(m);
    c_v.wait(lk, []{return left_descriptors.size() < 4 && right_descriptors.size() < 4;});
    left_descriptors.push(left);
    right_descriptors.push(right);
    lk.unlock();
    c_v.notify_all();
  }
}

int main (int argc, char** argv) {

  
  // we need the path name to 2010_03_09_drive_0019 as input argument
  if (argc<4) {
    cerr << "Usage: ./viso2 path/to/sequence/ num_frames result_file" << endl;
    return 1;
  }


  // We need to call this to set up global state for TensorFlow.
  //tensorflow::port::InitMain(argv[0], &argc, &argv);
  //if (argc > 1) {
  //  LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
  //  return -1;
  //}

  // sequence directory
  string dir = argv[1];
  int num_frames = atoi(argv[2]);
  string result_file = argv[3];
  string graph_path="trained/model_LR.pb";

  ofstream output_file;
  output_file.open(result_file.c_str());
  
  // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  Egomotion::parameters param;
  loadCalibParams(param, dir);
  param.match.refinement = 2;
  param.match.half_resolution = 0;
  param.match.use_initial_descriptor = false;
  param.match.sort = 1;
  param.ransac_iters = 10000;
  param.inlier_threshold = 2.0;
  param.reweighting = false;
  param.match.multi_stage = 1;
  param.bucket.max_features = 5;

  //param.match.nms_n = 1;
  //param.bucket.bucket_width = 2000;
  //param.bucket.bucket_height = 2000;
  //Ptr<Feature2D> detector = cv::ORB::create(20000);
  //Ptr<Feature2D> detector = cv::ORB::create(20000, 1.2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
  //Ptr<Feature2D> detector = cv::ORB::create(20000, 1.2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 10);
  shared_ptr<Detector> detector = DetectorFactory::constructDetector(DetectorFactory::Harris);

  ConvMatcher *matcher_conv   = new ConvMatcher(param.match, detector);
  // init visual odometry
  Egomotion viso_main(param, matcher_conv);
  //viso_main.setMatcher(matcher_conv);
  
  // current pose (this matrix transforms a point from the current
  // frame's camera coordinates to the first frame's camera coordinates)
  Matrix pose = Matrix::eye(4);

  srand(0);

  shared_ptr<ConvNetwork> extractor(new ConvNetwork(graph_path));

  std::thread t;
  for (int32_t i=0; i<num_frames; i++) {

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    // input file names
    //char base_name[256]; sprintf(base_name,"%06d.png",i);
    char base_name[256]; sprintf(base_name,"%06d",i);
    string left_img_file_name  = dir + "/image_0/" + base_name + ".png";
    string right_img_file_name = dir + "/image_1/" + base_name + ".png";
    
    // catch image read/write errors here
    try {

      // load left and right input image
      png::image< png::gray_pixel > left_img(left_img_file_name);
      png::image< png::gray_pixel > right_img(right_img_file_name);

      // image dimensions
      int32_t width  = left_img.get_width();
      int32_t height = left_img.get_height();

      if (i == 0) {
        t = std::thread(loadImages, extractor, dir, num_frames);
      }

      std::unique_lock<std::mutex> lk(m);
      c_v.wait(lk, []{return left_descriptors.size() > 0 && right_descriptors.size() > 0;});

      matcher_conv->pushBackFeatures(left_descriptors.front(), right_descriptors.front());
      left_descriptors.pop();
      right_descriptors.pop();

      lk.unlock();
      c_v.notify_all();

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
        cout << viso_main.getNumberOfInliers() << "/" << viso_main.getMatches().size() << endl;
        auto matches = viso_main.getMatches();
        writeMatches(matches, "../matches/conv/" + std::to_string(i));
      }
      else {
        final_T = Matrix::eye(4);
        cout << " ... failed! " << i << endl;
      }

      pose = pose * final_T;
      
      if (i % 25 == 0) {
        cout << dir << ": " << i << "/" << num_frames << endl;
        //if (i == 3) {
        //  return 0;
        //}
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
    } catch (const std::exception &e) {
      cerr << "ERROR: Couldn't read input files!" << endl;
      cout << e.what() << endl;
      return 1;
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    cout << "Duration: " << duration << endl;
  }
  
  t.join();
  // output
  cout << "Demo complete! Exiting ..." << endl;
  output_file.close();

  // exit
  return 0;
}

