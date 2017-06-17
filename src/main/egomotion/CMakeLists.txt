cmake_minimum_required (VERSION 2.8)


file( GLOB LIB_SOURCES ./*.cpp )
file( GLOB LIB_HEADERS ./*.h )

add_executable(egomotion ${LIB_SOURCES} ${LIB_HEADERS} )
target_link_libraries(egomotion egomotion_lib)
add_dependencies(egomotion egomotion_lib)



