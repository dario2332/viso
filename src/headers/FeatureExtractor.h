#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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


using namespace std;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

class FeatureExtractor {

public:
    FeatureExtractor(string model_path, string input_layer="input_left_forward", string output_layer="output");
    ~FeatureExtractor();
    
    int W, D, H;
    float* getFeature(int u, int v);
    void extractFeatures(string &image_path);
    void initDims(int w, int h, int d);
    vector<float> data;

private:
    std::unique_ptr<tensorflow::Session> session;
    string input_layer;
    string output_layer;

    Status LoadGraph(string graph_file_name, std::unique_ptr<tensorflow::Session>* session);
    Status ReadTensorFromImageFile(string file_name, const float input_mean, const float input_std,
                                                     std::vector<Tensor>* out_tensors);

};

#endif //FEATURE_EXTRACTOR_H
