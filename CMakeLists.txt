project(darknet_test)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/darknet/src)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/darknet)
add_executable(net_load src/net_load.cpp)
target_link_libraries(net_load darknet)
