cmake_minimum_required(VERSION 3.16)

project(paged_rolling_quantile LANGUAGES CXX)

# The library: two headers under include/, no dependencies.
add_library(rolling_quantile INTERFACE)
add_library(rq::rolling_quantile ALIAS rolling_quantile)
target_include_directories(rolling_quantile INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(rolling_quantile INTERFACE cxx_std_17)

# Everything below builds the tests and benchmarks of this repository and is skipped when the
# project is consumed with add_subdirectory().
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  return()
endif()

option(RQ_BUILD_TESTS "Build the tests" ON)
option(RQ_BUILD_BENCHMARKS "Build the benchmark programs" ON)
option(RQ_BUILD_EXAMPLES "Build the example program" ON)
option(RQ_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(RQ_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Settings shared by every program in this repository.
add_library(rq_project_options INTERFACE)
target_link_libraries(rq_project_options INTERFACE rolling_quantile)
target_include_directories(rq_project_options INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
if(MSVC)
  target_compile_options(rq_project_options INTERFACE /W4)
  if(RQ_WARNINGS_AS_ERRORS)
    target_compile_options(rq_project_options INTERFACE /WX)
  endif()
else()
  target_compile_options(rq_project_options INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
  if(RQ_WARNINGS_AS_ERRORS)
    target_compile_options(rq_project_options INTERFACE -Werror)
  endif()
  if(RQ_SANITIZE)
    set(rq_sanitizer_flags
      -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
    # -UNDEBUG keeps assert() active whatever the build type.
    target_compile_options(rq_project_options INTERFACE ${rq_sanitizer_flags} -UNDEBUG)
    target_link_options(rq_project_options INTERFACE ${rq_sanitizer_flags})
  endif()
endif()

if(RQ_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(RQ_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(RQ_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
