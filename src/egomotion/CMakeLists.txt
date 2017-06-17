cmake_minimum_required (VERSION 2.8)

file( GLOB LIB_SOURCES ./*.cpp )
file( GLOB LIB_HEADERS ./*.h )

add_library(egomotion_lib ${LIB_SOURCES} ${LIB_HEADERS} )
add_dependencies(egomotion_lib libviso2)
target_link_libraries(egomotion_lib libviso2 ${OpenCV_LIBS})

#include_directories(".")

