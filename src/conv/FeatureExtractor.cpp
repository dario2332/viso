#include "FeatureExtractor.h"

//std::unique_ptr<tensorflow::Session> FeatureExtractor::session;

FeatureExtractor::FeatureExtractor(string graph_path, string left_input_layer, string right_input_layer, 
                                   string left_output_layer, string right_output_layer) : 
                                   left_input_layer(left_input_layer), right_input_layer(right_input_layer), 
                                   left_output_layer(left_output_layer), right_output_layer(right_output_layer), W(0), D(0), H(0) {

//  static bool loaded = false;
//  if (!loaded) {
  // First we load and initialize the model.
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    //return -1;
  }
 // else loaded = true;

//  }
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status FeatureExtractor::ReadTensorFromImageFile(string file_name, 
                               const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";
  auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(input_name),
                                               file_name);
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 1;
  tensorflow::Output image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    image_reader = DecodeGif(root.WithOpName("gif_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, dims_expander, {0.0f}),
      {256.0f});

  Div(root.WithOpName(output_name), Sub(root, dims_expander, {0.39f}),
      {0.3f});
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status FeatureExtractor::LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  //tensorflow::SessionOptions options;
  //tensorflow::GPUOptions* gpu_options = new tensorflow::GPUOptions();
  //gpu_options->set_per_process_gpu_memory_fraction(0.4);
  //gpu_options->set_allow_growth(true);
  //options.config.set_allocated_gpu_options(gpu_options);

  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  //session->reset(tensorflow::NewSession(options));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}


void FeatureExtractor::extractFeatures(string &left_image_path, string &right_image_path,
                                                              shared_ptr<ImageDescriptor> &left, shared_ptr<ImageDescriptor> &right) {

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  float input_mean = 0;
  float input_std = 256;
  std::vector<Tensor> resized_tensors;
  Status read_tensor_status =
      ReadTensorFromImageFile(left_image_path, input_mean, input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    //return -1;
  }
  const Tensor resized_tensor_L = resized_tensors[0];

  resized_tensors.clear();
  read_tensor_status =
      ReadTensorFromImageFile(right_image_path, input_mean, input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    //return -1;
  }
  const Tensor resized_tensor_R = resized_tensors[0];
  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{left_input_layer, resized_tensor_L}, {right_input_layer, resized_tensor_R}},
                                   {left_output_layer, right_output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    //return -1;
  }

  left = shared_ptr<ImageDescriptor> (new ImageDescriptor(W, H, D, left_image_path));
  right = shared_ptr<ImageDescriptor> (new ImageDescriptor(W, H, D, right_image_path));

  tensorflow::TTypes<float>::Flat final_tensor_L = outputs[0].flat<float>();
  tensorflow::TTypes<float>::Flat final_tensor_R = outputs[1].flat<float>();
  std::copy (&final_tensor_L(0), &final_tensor_L(W*H*D-1), left->data.begin());
  std::copy (&final_tensor_R(0), &final_tensor_R(W*H*D-1), right->data.begin());
}


//u-W ,  v-H
float* ImageDescriptor::getFeature(int u, int v) {
  return &data.at(w * d * v + d * u);
}

FeatureExtractor::~FeatureExtractor() {
}

void FeatureExtractor::initDims(int w, int h, int d) {
  W = w; H = h; D = d;
  //data.resize(W*H*D);
}
