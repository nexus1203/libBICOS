cmake_minimum_required(VERSION 3.18)
if(POLICY CMP0115)
  cmake_policy(SET CMP0115 OLD)
endif()
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 OLD)
endif()
project(libBICOS VERSION 2.2.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BICOS_IMPLEMENTATION "Implementation to use (CPU or CUDA)" "CPU")
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build test programs" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)

# Set implementation flags
if(BICOS_IMPLEMENTATION STREQUAL "CPU")
    add_definitions(-DBICOS_CPU)
elseif(BICOS_IMPLEMENTATION STREQUAL "CUDA")
    add_definitions(-DBICOS_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Use pre-generated CUDA source
    set(CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/cuda.cu
        src/impl/cuda/cutil.cu
    )
    # CUDA compilation flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-deprecated-enum-enum-conversion")
else()
    message(FATAL_ERROR "Unsupported implementation: ${BICOS_IMPLEMENTATION}")
endif()

# Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_definitions(-DBICOS_DEBUG)
endif()

# Find dependencies
find_package(OpenCV REQUIRED)

# Set include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Configure header file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/BICOS/config.hpp
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Library sources
set(BICOS_SOURCES
    src/lib.cpp
    src/exception.cpp
    src/formatable.cpp
)

# Add implementation-specific sources
if(BICOS_IMPLEMENTATION STREQUAL "CPU")
    list(APPEND BICOS_SOURCES src/impl/cpu.cpp)
elseif(BICOS_IMPLEMENTATION STREQUAL "CUDA")
    list(APPEND BICOS_SOURCES ${CUDA_SOURCES})
endif()

# Create library
add_library(BICOS ${BICOS_SOURCES})
target_link_libraries(BICOS PUBLIC ${OpenCV_LIBS})
set_target_properties(BICOS PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Python bindings using ctypes
add_library(pybicos_c SHARED src/pybicos_c.cpp)
target_link_libraries(pybicos_c PRIVATE BICOS ${OpenCV_LIBS})
set_target_properties(pybicos_c PROPERTIES
    PREFIX ""
    OUTPUT_NAME "pybicos_c"
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install targets
install(TARGETS BICOS pybicos_c
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(FILES 
    include/common.hpp 
    include/match.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/include/BICOS/config.hpp
    DESTINATION include/BICOS
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    # Add test targets here
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    # Add benchmark targets here
endif()
