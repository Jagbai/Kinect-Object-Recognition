cmake_minimum_required(VERSION 2.8)

PROJECT(VIExamples)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Executables")


find_package(OpenCV REQUIRED)

# Modify the following line to add new files to your project 
add_executable(opencvexample Prediction.cpp freenect-playback-wrapper.h freenect-playback-wrapper.cpp )
 
target_link_libraries(opencvexample ${OpenCV_LIBS})
