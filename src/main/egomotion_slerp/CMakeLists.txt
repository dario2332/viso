cmake_minimum_required (VERSION 2.8)


file( GLOB LIB_SOURCES ./*.cpp )
file( GLOB LIB_HEADERS ./*.h )

add_executable(egomotion_slerp ${LIB_SOURCES} ${LIB_HEADERS} )
target_link_libraries(egomotion_slerp egomotion_lib)
add_dependencies(egomotion_slerp egomotion_lib)



