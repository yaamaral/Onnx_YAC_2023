#meuyara
cmake_minimum_required(VERSION 3.0.0)
project(onnxruntime VERSION 0.1.0)

include(CTest)
enable_testing()
set(CMAKE_CXX_STANDARD 11)

#find_package(oonnxruntime/1.15.1 REQUIRED)
find_package(OpenCV)
#message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
#message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

include_directories(/opt/homebrew/Cellar/onnxruntime/1.15.1/include/)
include_directories(${OpenCV_INCLUDE_DIRS})

#add_executable(onnx-cpp susy-onnx.cpp)

add_executable(onnx-cpp OnnxRuntimeResNet.cpp)

target_link_libraries(onnx-cpp /opt/homebrew/Cellar/onnxruntime/1.15.1/lib/libonnxruntime.1.15.1.dylib)

#target_link_libraries(onnx-cpp /opt/homebrew/opt/mbedtls/lib/libmbedcrypto.14.dylib)#

target_link_libraries(onnx-cpp ${OpenCV_LIBS})


#set( 
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
#/opt/homebrew/Cellar/onnxruntime/1.15.1/lib/libonnxruntime.1.15.1.dylib -rpath /opt/homebrew/Cellar/onnxruntime/1.15.1/lib/libonnxruntime.1.15.1.dylib/lib/ -o ./$(PROGRAM)
