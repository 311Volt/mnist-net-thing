cmake_minimum_required(VERSION 3.0.0)
project(mnist_net_thing VERSION 0.1.0)


FILE(GLOB_RECURSE MNIST_NET_SOURCES "src/*.cpp")

if(WIN32)
	set(CMAKE_FIND_LIBRARY_SUFFIXES .a .dll .dll.a)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
endif()

find_library(LIBALLEG allegro_monolith "deps/lib")
find_library(LIBFFTW fftw3f-3 "deps/lib")

add_executable(mnist_net_thing ${MNIST_NET_SOURCES})

target_link_libraries(mnist_net_thing ${LIBALLEG} ${LIBFFTW})
target_include_directories(mnist_net_thing PUBLIC "deps/include")

add_custom_command(TARGET mnist_net_thing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_command(TARGET mnist_net_thing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/deps/dll ${CMAKE_CURRENT_BINARY_DIR}
)
