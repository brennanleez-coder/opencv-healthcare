cmake_minimum_required(VERSION 3.10)

project(OpenCvHealthcare)

include_directories(src/include)
file(GLOB SOURCES "src/*.cpp" "src/lib/**/*.cpp")

add_executable(main ${SOURCES})


#  Add the pybind11 library
add_subdirectory(external/pybind11)

target_link_libraries(main pybind11::embed)
