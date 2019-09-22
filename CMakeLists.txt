cmake_minimum_required (VERSION 3.10)

option (gorpho_build_tests "Build gorpho tests" OFF)
option (gorpho_download_googletest "Download and build googletest as part of test config" ON)

enable_language (CUDA)

project ("gorpho" VERSION 0.1.0
		          DESCRIPTION "Header-only library for fast GPU morphology with CUDA."
				  LANGUAGES "CUDA")

add_library (${PROJECT_NAME} INTERFACE)
target_include_directories (${PROJECT_NAME} INTERFACE
	"${${PROJECT_NAME}_SOURCE_DIR}/lib" "${${PROJECT_NAME}_SOURCE_DIR}/lib/cudablockproc")
# This is to make autocorrect and intellisense work
target_include_directories (${PROJECT_NAME} INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if (gorpho_build_tests)
	enable_testing () # This must be set in the root directory so the tests are discovered
	add_subdirectory ("test")
endif ()