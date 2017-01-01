project(darknet_test)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/darknet/src)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/darknet)

# CUDA
find_package(CUDA)
if (CUDA_FOUND)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
  add_definitions(-DGPU)

  # cuDNN
  find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    PATHS ${CUDA_INCLUDE_DIRS} /usr/local/cuda/include
    PATH_SUFFIXES include)
  find_library(CUDNN_LIBRARY
    NAMES cudnn
    PATHS ${CUDNN_LIB_DIR} /usr/local/cuda/lib64)
  if (EXISTS ${CUDNN_INCLUDE_DIR} AND EXISTS ${CUDNN_LIBRARY})
    list(APPEND LIBRARIES ${CUDNN_LIBRARY})
    include_directories(${CUDNN_INCLUDE_DIR})
    add_definitions(-DCUDNN)
  endif()
endif()

add_executable(net_load src/net_load.cpp)
target_link_libraries(net_load ${OpenCV_LIBS} darknet)
add_executable(test_detector src/test_detector.cpp)
target_link_libraries(test_detector ${OpenCV_LIBS} darknet)
