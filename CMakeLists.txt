project(darknet_test)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/darknet/src)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/darknet)

include(cmake/cuda_darknet.cmake)
cudacheck()

add_executable(net_load src/net_load.cpp)
target_link_libraries(net_load ${OpenCV_LIBS} darknet)
add_executable(test_detector src/test_detector.cpp)
target_link_libraries(test_detector ${OpenCV_LIBS} darknet)
add_executable(video_detector src/video_detector.cpp)
target_link_libraries(video_detector ${OpenCV_LIBS} darknet)
