#meuYara
// OnnxRuntimeResNet.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <iostream>
#include "Helpers.cpp"

int main()
{
	Ort::Env env;
	Ort::RunOptions runOptions;


    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannels * height * width;

     const std::string imageFile = "/Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/assets/dog.png";
     const std::string labelFile = "/Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/assets/imagenet_classes.txt";
     std::string modelPath = "/Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/assets/resnet50-v2-7.onnx";
    // resnet50-v2-7.onnx  or resnet50v2.onnx

    //load labels
    std::vector<std::string> labels = loadLabels(labelFile);
    if (labels.empty()) {
        std::cout << "Failed to load labels: " << labelFile << std::endl;
        return 1;
    }

    // load image
    const std::vector<float> imageVec = loadImage(imageFile);
    if (imageVec.empty()) {
        std::cout << "Failed to load image: " << imageFile << std::endl;
        return 1;
    }

    if (imageVec.size() != numInputElements) {
        std::cout << "Invalid image format. Must be 224x224 RGB image." << std::endl;
        return 1;
    }

    // create session
    Ort::SessionOptions session_opt{ nullptr };
    
    Ort::Session session = Ort::Session(env, modelPath.c_str(),session_opt );

    // define shape
    const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
    const std::array<int64_t, 2> outputShape = { 1, numClasses };

    // define array
    std::array<float, numInputElements> input;
    std::array<float, numClasses> results;

     // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // copy image data to input array
    std::vector<float> imageVec(3*224*224);
    std::fill(imageVec.begin(), imageVec.end(), 0);
   // for (char i: imageVec)
   // std::cout << i << ' ';
    std::copy(imageVec.begin(), imageVec.end(), input.begin());

    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const char* const* inputNames = (const char* const*)inputName.get(); 
    const char* const* outputNames = (const char* const*)outputName.get(); 
     // run inference
    try {
        session.Run(runOptions, inputNames, &inputTensor, 1, outputNames, &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
    //char* inputName = 'i';
    //char* outputName = 'o';
    //const std::array<const char*, 1> inputNames = { inputName };
    //const std::array<const char*, 1> outputNames = { outputName };
    //ort_alloc.Free(inputName);
    //ort_alloc.Free(outputName);
}
