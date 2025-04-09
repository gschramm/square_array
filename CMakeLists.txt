cmake_minimum_required(VERSION 3.18)
project(PARALLELPROJ LANGUAGES C CXX)

include(CheckLanguage)
check_language(CUDA)

# Derive version from Git or folder name
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git describe --tags --long --dirty
        OUTPUT_VARIABLE GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "v?([0-9]+\\.[0-9]+\\.[0-9]+)" PROJECT_VERSION "${GIT_DESCRIBE}")
    string(REGEX REPLACE "^v" "" PROJECT_VERSION "${PROJECT_VERSION}")
else()
    get_filename_component(FOLDER_NAME ${CMAKE_SOURCE_DIR} NAME)
    string(REGEX MATCH "([0-9]+\\.[0-9]+\\.[0-9]+)" PROJECT_VERSION "${FOLDER_NAME}")
    if(NOT PROJECT_VERSION)
        set(PROJECT_VERSION "0.0.0")
    endif()
endif()

string(REGEX MATCHALL "[0-9]+" VERSION_PARTS "${PROJECT_VERSION}")
list(GET VERSION_PARTS 0 PROJECT_VERSION_MAJOR)
list(GET VERSION_PARTS 1 PROJECT_VERSION_MINOR)
list(GET VERSION_PARTS 2 PROJECT_VERSION_PATCH)

message(STATUS "Building version ${PROJECT_VERSION}")

option(DEBUG "Enable debug printing" OFF)
if(DEBUG)
    add_compile_definitions(DEBUG)
endif()

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    add_library(parallelproj SHARED joseph3d_fwd.cu)
    target_link_libraries(parallelproj PRIVATE CUDA::cudart)

    add_executable(main main.cu)
    target_include_directories(main PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(main PRIVATE parallelproj CUDA::cudart)
else()
    find_package(OpenMP REQUIRED)
    add_library(parallelproj SHARED joseph3d_fwd.cpp)
    target_link_libraries(parallelproj PRIVATE OpenMP::OpenMP_CXX)
    add_executable(main main.cpp)
    target_link_libraries(main PRIVATE parallelproj)
endif()
