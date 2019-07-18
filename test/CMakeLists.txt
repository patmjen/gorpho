cmake_minimum_required (VERSION 3.10)

add_executable (tests volume_tests.cu view_tests.cu general_morph_tests.cu flat_linear_morph_tests.cu)
target_compile_options (tests PUBLIC "--expt-relaxed-constexpr" "-lineinfo")
target_link_libraries (tests gtest_main)
add_test (NAME all_tests COMMAND tests)