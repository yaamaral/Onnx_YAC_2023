# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.27.4/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.27.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/build

# Utility rule file for ExperimentalStart.

# Include any custom commands dependencies for this target.
include CMakeFiles/ExperimentalStart.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ExperimentalStart.dir/progress.make

CMakeFiles/ExperimentalStart:
	/opt/homebrew/Cellar/cmake/3.27.4/bin/ctest -D ExperimentalStart

ExperimentalStart: CMakeFiles/ExperimentalStart
ExperimentalStart: CMakeFiles/ExperimentalStart.dir/build.make
.PHONY : ExperimentalStart

# Rule to build all files generated by this target.
CMakeFiles/ExperimentalStart.dir/build: ExperimentalStart
.PHONY : CMakeFiles/ExperimentalStart.dir/build

CMakeFiles/ExperimentalStart.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ExperimentalStart.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ExperimentalStart.dir/clean

CMakeFiles/ExperimentalStart.dir/depend:
	cd /Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet /Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet /Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/build /Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/build /Users/yaradoamaralcoutinho/Projects/Onnx_YAC/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/build/CMakeFiles/ExperimentalStart.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/ExperimentalStart.dir/depend

