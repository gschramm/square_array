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
    set(LIB_SOURCES
        src/joseph3d_fwd.cu
        src/joseph3d_back.cu
        src/cuda_utils.cu
    )

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    add_library(parallelproj SHARED ${LIB_SOURCES})
    target_include_directories(parallelproj PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(parallelproj PRIVATE CUDA::cudart)

    set(TEST_SOURCES
        tests/test_nontof_fwd_back.cu
    )
else()
    set(LIB_SOURCES
        src/joseph3d_fwd.cpp
        src/joseph3d_back.cpp
    )

    find_package(OpenMP REQUIRED)
    add_library(parallelproj SHARED ${LIB_SOURCES})
    target_include_directories(parallelproj PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(parallelproj PRIVATE OpenMP::OpenMP_CXX)

    set(TEST_SOURCES
        tests/test_nontof_fwd_back.cpp
    )
endif()


################################################################################
### BUILD TESTS ################################################################
################################################################################

enable_testing()

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    if(CMAKE_CUDA_COMPILER)
        target_link_libraries(${TEST_NAME} PRIVATE parallelproj CUDA::cudart)
    else()
        target_link_libraries(${TEST_NAME} PRIVATE parallelproj OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
