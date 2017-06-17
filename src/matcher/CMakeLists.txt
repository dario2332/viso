cmake_minimum_required (VERSION 2.8)

file( GLOB LIB_SOURCES ./ConvMatcher.cpp ./CustomDetectorConvMatcher.cpp)
file( GLOB LIB_HEADERS ./ConvMatcher.h ./CustomDetectorConvMatcher.h )

add_library(matcher ${LIB_SOURCES} ${LIB_HEADERS} )
add_dependencies(matcher libviso2)
target_link_libraries(matcher openblas)



