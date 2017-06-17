
cmake_minimum_required (VERSION 2.8)


file( GLOB LIB_SOURCES ./*.cpp )
file( GLOB LIB_HEADERS ./*.h )

add_library(libviso2 ${LIB_SOURCES} ${LIB_HEADERS} )
target_link_libraries(libviso2 png)


