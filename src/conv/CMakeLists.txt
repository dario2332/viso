cmake_minimum_required (VERSION 2.8)

file( GLOB LIB_SOURCES ./*.cpp )
file( GLOB LIB_HEADERS ./*.h )

add_library(conv_features ${LIB_SOURCES} ${LIB_HEADERS} )
target_link_libraries(conv_features ${TensorFlow_LIBRARIES})

#include_directories(".")

