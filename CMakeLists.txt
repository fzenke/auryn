cmake_minimum_required (VERSION 2.6)
project (Auryn)

# The version number.
set (Auryn_VERSION_MAJOR 0)
set (Auryn_VERSION_MINOR 6)

# Important GCC Compiler flags for Auryn's performance
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math")
# add_definitions(-Wall) 
# Display flags
message("CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")

# Required external libraries (Boost and and MPI implementation)
FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(Boost 1.5 COMPONENTS mpi serialization program_options date_time REQUIRED)

INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES(src)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(examples)

