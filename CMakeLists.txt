#cmake_minimum_required(VERSION 3.0...3.26)
cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(hnswlib
    LANGUAGES CXX)

# Define your Conda environment root directory
set(MY_CONDA_ENV "/export/data/zgongae/conda/envs/model")

# 1. Tell CMake to prioritize searching packages and libraries in Conda environment
list(APPEND CMAKE_PREFIX_PATH "${MY_CONDA_ENV}")
list(APPEND CMAKE_LIBRARY_PATH "${MY_CONDA_ENV}/lib")
list(APPEND CMAKE_INCLUDE_PATH "${MY_CONDA_ENV}/include")

# 2. If the above doesn't work, here's a forced patch for NVTX
# PyTorch sometimes depends on FindCUDA, which may not find nvToolsExt in Conda
# We manually specify its location (usually under lib)
find_library(LIBNVTOOLSEXT_LIBRARY 
    NAMES nvToolsExt 
    PATHS "${MY_CONDA_ENV}/lib"
    NO_DEFAULT_PATH
)

# If found, explicitly set it for variables that PyTorch might use (this is a safety measure)
if(LIBNVTOOLSEXT_LIBRARY)
    message(STATUS "Found nvToolsExt: ${LIBNVTOOLSEXT_LIBRARY}")
    # Some versions of Torch cmake use this variable
    set(CUDA_nvToolsExt_LIBRARY ${LIBNVTOOLSEXT_LIBRARY}) 
else()
    message(WARNING "Could not find libnvToolsExt.so in conda env. Linking might fail.")
endif()

set(Torch_DIR "/export/data/zgongae/conda/envs/model/lib/python3.11/site-packages/torch/share/cmake/Torch") 
find_package(Torch REQUIRED)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)




add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

target_compile_features(hnswlib INTERFACE cxx_std_17)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 11)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()

    # examples
    # Version: PyTorch enabled version
    add_executable(example_search examples/cpp/example_search.cpp)
    target_compile_definitions(example_search PRIVATE USE_PYTORCH)
    target_include_directories(example_search PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(example_search PRIVATE hnswlib::hnswlib ${TORCH_LIBRARIES})
    set_property(TARGET example_search PROPERTY CXX_STANDARD 17)

    #add_executable(preparedata examples/cpp/preparedata.cpp)

    #add_executable(example_epsilon_search examples/cpp/example_epsilon_search.cpp)
    #target_link_libraries(example_epsilon_search hnswlib)

    #add_executable(example_multivector_search examples/cpp/example_multivector_search.cpp)
    #target_link_libraries(example_multivector_search hnswlib)

    #add_executable(example_filter examples/cpp/example_filter.cpp)
    #target_link_libraries(example_filter hnswlib)

    #add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    #target_link_libraries(example_replace_deleted hnswlib)

    #add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    #target_link_libraries(example_mt_search hnswlib)

    #add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    #target_link_libraries(example_mt_filter hnswlib)

    #add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)
    #target_link_libraries(example_mt_replace_deleted hnswlib)

    # tests
    #add_executable(multivector_search_test tests/cpp/multivector_search_test.cpp)
    #target_link_libraries(multivector_search_test hnswlib)

    #add_executable(epsilon_search_test tests/cpp/epsilon_search_test.cpp)
    #target_link_libraries(epsilon_search_test hnswlib)

    #add_executable(test_updates tests/cpp/updates_test.cpp)
    #target_link_libraries(test_updates hnswlib)

    #add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    #target_link_libraries(searchKnnCloserFirst_test hnswlib)

    #add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    #target_link_libraries(searchKnnWithFilter_test hnswlib)

    #add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    #target_link_libraries(multiThreadLoad_test hnswlib)

    #add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)
    #target_link_libraries(multiThread_replace_test hnswlib)

    #add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    #target_link_libraries(main hnswlib)

    #add_executable(audio examples/cpp/search_audio_hnsw.cpp)
    #target_link_libraries(audio hnswlib)

    #target_link_libraries(example_pytorch_search PRIVATE hnswlib::hnswlib torch)
endif()
