# The tests compare floating-point results bit for bit, so the two sides of a comparison must be
# compiled to the same operations: keep the compiler from fusing a multiply and an add into one
# FMA in some places and not in others.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-ffp-contract=off RQ_HAS_FP_CONTRACT_OFF)

function(rq_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE rq_project_options)
  if(RQ_HAS_FP_CONTRACT_OFF)
    target_compile_options(${name} PRIVATE -ffp-contract=off)
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

rq_add_test(test_paged_multiset)
rq_add_test(test_definition)
rq_add_test(test_differential)
rq_add_test(test_reference)
rq_add_test(test_no_allocation)
