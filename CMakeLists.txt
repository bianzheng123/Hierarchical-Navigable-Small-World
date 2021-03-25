cmake_minimum_required(VERSION 2.6)
project(multiple_hnsw)

include_directories("${PROJECT_BINARY_DIR}")


add_executable(hnsw hnsw.cpp)
#add_executable(cascade_hnsw cascade_hnsw.cpp)
add_executable(bruteforce bruteforce.cpp)

add_executable(test test.cpp)
add_executable(vecs test_vector.cpp)
add_executable(shuffle test_shuffle.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    SET(CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize")
endif ()
