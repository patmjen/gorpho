cmake_minimum_required (VERSION 3.10)

option (gorpho_download_googletest "Download and build googletest as part of CMake config" ON)

enable_language (CUDA)

project ("gorpho" VERSION 0.1.0
		          DESCRIPTION "Header-only library for fast GPU morphology with CUDA."
				  LANGUAGES "CUDA")

add_library (${PROJECT_NAME} INTERFACE)
target_include_directories (${PROJECT_NAME} INTERFACE
	"${${PROJECT_NAME}_SOURCE_DIR}/lib" "${${PROJECT_NAME}_SOURCE_DIR}/lib/cudablockproc")
# This is to make autocorrect and intellisense work
target_include_directories (${PROJECT_NAME} INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory ("test" EXCLUDE_FROM_ALL)