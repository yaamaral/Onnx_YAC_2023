// MeuYara OnnxRuntimeResNet.cpp : This file contains the 'main' function. Program execution begins and ends there. https://www.youtube.com/watch?v=imjqRdsm2Qw

#include </opt/homebrew/Cellar/onnxruntime/1.15.1/include/onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <iostream>

#include "Helpers.cpp"

    int main()
{
// Create environment
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);

    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannels * height * width;

    
    const std::string imageFile = "/Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/assets/dog.png";
    const std::string labelFile = "/Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/assets/imagenet_classes.txt";
    //std::string
        std::string  modelPath = "/Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/assets/resnet50-v2-7.onnx";
       // resnet50v2.onnx or resnet50-v2-7.onnx
    // load labels
    std:: vector<std::string> labels = loadLabels(labelFile);
    if (labels.empty ()) {
        std:: cout << "Failed to load labels: " << labelFile << std:: endl;
        return 1;
    }

    // load image
    const std:: vector<float> imageVec = loadImage(imageFile);
    if (imageVec.empty() ) {
        std::cout << "Failed to load image: " << imageFile << std::endl;
        return 1;
    }

    if (imageVec.size() != numInputElements){

        std::cout << "Invalid image format. Must be 224x224 RGB image. " << std::endl;
        return 1;
    }
     
    //create session
    session = Ort::Session (env, modelPath.c_str(), Ort::SessionOptions{ nullptr });

    //define shape
    const std::array<int64_t, 4> inputShape{1, numChannels, height, width };
    const std::array<int64_t, 2> outputShape = {1, numClasses};

    //define array
    std::array<float, numInputElements> input;
    std::array<float, numClasses> results;

    // define  Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // tirando std::cout << "Input data : " << input.data() << " size : " << input.size() << " input shape data : " << inputShape.data() << " size : " << inputShape.size() << std::endl;
    
    
    // print input node types
    const size_t num_input_nodes = session.GetInputCount();

    // tirando std::cout << "num_input_nodes " << num_input_nodes << std::endl ;

    auto type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    auto type = tensor_info.GetElementType();
    //tirando std::cout << "Input : type = " << type << std::endl;

    // print input shapes/dims
    auto input_node_dims = tensor_info.GetShape();
    
    //tirando std::cout << "Input : num_dims = " << input_node_dims.size() << std::endl;
    
    auto input_tensor_size =1;


    for (size_t j = 0; j < input_node_dims.size(); j++) {
      std::cout << "Input : dim[" << j << "] = " << input_node_dims[j] << std::endl;
      input_tensor_size *=input_node_dims[j];
   }

    //tirando std::cout << "input_node_dims data = " << input_node_dims.data() << " size = " << input_node_dims.size() << std::endl;
    //tirando std::cout << "input data = " << input.data() << std::endl;
    //tirando std::cout << " input_tensor_size " << input_tensor_size << std::endl ;
    
    
    
    
    
    
    //copy image data to input array
    std:: copy(imageVec.begin(), imageVec.end(), input.begin());


    
    //define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char *, 1> inputNames = {inputName.get()};
    const std::array<const char *, 1> outputNames = {outputName.get()};
    
    std::cout << "\n" << *outputName.get() << std::endl;
    std::cout << "\n" << *outputNames.data() << std::endl;

    
    //run inference
    
    try{
        session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
    
    std::cout << results.size() << std::endl;
    
    
    // results
    std::vector<std::pair<size_t, float>> indexValuePairs;
    for (size_t i =0 ; i < results.size(); i++){
        indexValuePairs.emplace_back(i, results[i]);
        
        std::cout << i << ": " << labels [i] << ";  " << results [i]  << "; "  << "\n";
        std::cout << "\n\n";
    }
   
    // show results labels and score
    //for (size_t i = 0; i < 999; i++) {
     //  const auto& result = indexValuePairs[i];
    //   std::cout << i + 1 << " : " << labels[result.first] << "  " << labels[result.second] << ": " << std::endl;
        
    // }
}
