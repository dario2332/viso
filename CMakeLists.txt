# project
cmake_minimum_required (VERSION 2.6)
project (libviso2)

# directories
set (LIBVISO2_SRC_DIR src)

# include directory
include_directories("${LIBVISO2_SRC_DIR}")

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-std=c++11 -msse3 -O3")

# sources
#FILE(GLOB LIBVISO2_SRC_FILES "src/lib/*.cpp")
file( GLOB LIB_SOURCES src/lib/*.cpp )
file( GLOB LIB_HEADERS src/lib/headers/*.h )
add_library( viso_lib ${LIB_SOURCES} ${LIB_HEADERS} )
# make release version
set(CMAKE_BUILD_TYPE Release)

find_package( OpenCV REQUIRED )
# demo program
#add_executable(viso2 ${LIBVISO2_SRC_FILES})
add_executable(viso2 src/main.cpp )
add_executable(viso2_slerp src/main_slerp.cpp )
target_link_libraries (viso2 viso_lib png ${OpenCV_LIBS} )
target_link_libraries (viso2_slerp viso_lib png ${OpenCV_LIBS} )

include_directories(${DIR}/headers src/lib/headers /opt/local/include /usr/local/boost_1_62_0)
link_directories (${DIR}/src/utils /opt/local/lib)
