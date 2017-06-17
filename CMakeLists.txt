# project
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

cmake_minimum_required (VERSION 2.8)
project (libviso2)

# Add modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

# directories
set (LIBVISO2_SRC_DIR src)

# include directory
include_directories("${LIBVISO2_SRC_DIR}")

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3")
#SET(CMAKE_CXX_FLAGS "-std=c++11 -g")

# sources
#FILE(GLOB LIBVISO2_SRC_FILES "src/lib/*.cpp")
#file( GLOB LIB_SOURCES src/lib/*.cpp )
#file( GLOB LIB_HEADERS src/lib/headers/*.h )
#add_library( viso_lib ${LIB_SOURCES} ${LIB_HEADERS} )
#add_dependencies(viso_lib Eigen Protobuf)

# make release version
set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(ENV{OPENBLAS_NUM_THREADS} 4)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS iostreams)
find_package(TensorFlow REQUIRED)

include(Protobuf)
include(Eigen)

# set variables for external dependencies
set(EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external" CACHE PATH "Location where external dependencies will installed")
set(DOWNLOAD_LOCATION "${EXTERNAL_DIR}/src" CACHE PATH "Location where external projects will be downloaded")
mark_as_advanced(EXTERNAL_DIR DOWNLOAD_LOCATION)

#add_executable(viso2 src/main.cpp )
#add_executable(viso2_slerp src/main_slerp.cpp )
#add_executable(viso2_features src/main_features.cpp src/ConvMatcher.cpp src/FeatureExtractor.cpp src/CustomDetectorConvMatcher.cpp)
#add_executable(evaluate_sad_metric src/evaluate_sad_metric.cpp src/ConvMatcher.cpp src/FeatureExtractor.cpp)
#add_executable(evaluate_cnn_metric src/evaluate_cnn_metric.cpp src/ConvMatcher.cpp src/FeatureExtractor.cpp)
#
#add_executable(orb_viso src/main_orb_detector src/OrbMatcher.cpp src/FeatureExtractor.cpp)
#
#add_dependencies(orb_viso Eigen Protobuf)
#add_dependencies(viso2_features Eigen Protobuf)
#
#target_link_libraries (viso2 viso_lib png ${OpenCV_LIBS} )
#target_link_libraries (viso2_slerp viso_lib png ${OpenCV_LIBS} )
#target_link_libraries (viso2_features viso_lib png pthread openblas ${Boost_LIBRARIES} ${OpenCV_LIBS} ${TensorFlow_LIBRARIES})
#target_link_libraries (evaluate_sad_metric viso_lib png openblas ${Boost_LIBRARIES} ${OpenCV_LIBS} ${TensorFlow_LIBRARIES})
#target_link_libraries (evaluate_cnn_metric viso_lib png openblas ${Boost_LIBRARIES} ${OpenCV_LIBS} ${TensorFlow_LIBRARIES})
#target_link_libraries (orb_viso viso_lib openblas ${Boost_LIBRARIES} ${OpenCV_LIBS} ${TensorFlow_LIBRARIES})

include_directories(${DIR}/headers src/egomotion/ src/libviso src/conv src/matcher /opt/local/include /usr/local/boost_1_62_0 ${TensorFlow_INCLUDE_DIRS} ${EXTERNAL_DIR}/include)
link_directories (${DIR}/src/utils /usr/lib /usr/local/lib ${PROJECT_SOURCE_DIR}/lib /opt/local/lib /usr/local/boost_1_62_0/stage/lib)

add_subdirectory(src/libviso)
add_subdirectory(src/egomotion)
add_subdirectory(src/matcher)
add_subdirectory(src/conv)
add_subdirectory(src/main)
#add_subdirectory(src/)
#add_subdirectory(src/libviso)
#add_subdirectory(src/libviso)

