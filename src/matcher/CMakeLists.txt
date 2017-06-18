cmake_minimum_required (VERSION 2.8)

file( GLOB LIB_SOURCES ./ConvMatcher.cpp ./FeatureMatcher.cpp ./Detector.cpp)
file( GLOB LIB_HEADERS ./ConvMatcher.h ./FeatureMatcher.h ./Detector.h)

add_library(matcher ${LIB_SOURCES} ${LIB_HEADERS} )
add_dependencies(matcher libviso2 conv_features)
target_link_libraries(matcher openblas conv_features)



