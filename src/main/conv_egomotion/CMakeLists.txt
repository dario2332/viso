cmake_minimum_required (VERSION 2.8)


file( GLOB LIB_SOURCES ./*.cpp )
file( GLOB LIB_HEADERS ./*.h )

add_executable(conv_egomotion ${LIB_SOURCES} ${LIB_HEADERS} )
target_link_libraries(conv_egomotion egomotion_lib matcher conv_features)
add_dependencies(conv_egomotion egomotion_lib matcher conv_features)



