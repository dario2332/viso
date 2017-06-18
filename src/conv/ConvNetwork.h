#ifndef CONV_H
#define CONV_H

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

#include "ImageDescriptor.h"

using namespace std;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

class ConvNetwork {

public:
    ConvNetwork(string model_path, string left_input_layer="input_left_forward", string right_input_layer="input_right_forward", 
                                        string left_output_layer="output_L", string right_output_layer="output_R");
    ~ConvNetwork();
    
    void extractFeatures(string &left_image_path, string &right_image_path, shared_ptr<ImageDescriptor> &left, shared_ptr<ImageDescriptor> &right);

private:
    std::unique_ptr<tensorflow::Session> session;
    string left_input_layer;
    string left_output_layer;
    string right_input_layer;
    string right_output_layer;

    Status LoadGraph(string graph_file_name, std::unique_ptr<tensorflow::Session>* session);
    Status ReadTensorFromImageFile(string file_name, const float input_mean, const float input_std,
                                                     std::vector<Tensor>* out_tensors);

};

#endif //CONV_H
