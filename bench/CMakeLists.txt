# The benchmark records how it was compiled in the header of its output.
string(TOUPPER "${CMAKE_BUILD_TYPE}" rq_build_type)
string(STRIP "-std=c++17 ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${rq_build_type}}" rq_bench_flags)

add_executable(bench_rolling_quantile bench_rolling_quantile.cpp)
target_link_libraries(bench_rolling_quantile PRIVATE rq_project_options)
target_compile_definitions(bench_rolling_quantile PRIVATE "RQ_BENCH_FLAGS=\"${rq_bench_flags}\"")

add_executable(memory_footprint memory_footprint.cpp)
target_link_libraries(memory_footprint PRIVATE rq_project_options)

# A short run of each program as part of the test suite, so that the compilers' warnings and the
# sanitizers cover them as well.
if(RQ_BUILD_TESTS)
  add_test(NAME bench_smoke
    COMMAND bench_rolling_quantile --windows 16,300,5000 --reps 2 --pass-seconds 0.01
            --min-samples 500 --max-samples 2000)
  add_test(NAME memory_footprint_smoke COMMAND memory_footprint 16 3000)
endif()
