cmake_minimum_required(VERSION 3.10)

project(OpenCvHealthcare)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    RESULT_VARIABLE NUMPY_FIND_RESULT
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT NUMPY_INCLUDE_DIR)
    message(FATAL_ERROR "NumPy include directory not found")
else()
    message(STATUS "NumPy include directory found at ${NUMPY_INCLUDE_DIR}")
endif()

# find_package(pybind11 REQUIRED)

include_directories(
    ${NUMPY_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    # ${pybind11_INCLUDE_DIRS}
    src/include
    src/cython/include
)

file(GLOB SOURCES "src/*.cpp" "src/lib/**/*.cpp" "src/algo/**/*.cpp")
add_executable(main ${SOURCES})


#  Add the pybind11 library
add_subdirectory(external/pybind11)

target_link_libraries(main pybind11::embed Python3::Python Python3::Module)
