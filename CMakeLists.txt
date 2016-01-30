cmake_minimum_required (VERSION 2.6)
project (Auryn)

# The version number.
set (Auryn_VERSION_MAJOR 0)
set (Auryn_VERSION_MINOR 7)

# Important GCC Compiler flags for Auryn's performance
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math -pipe")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pedantic")

# Display flags (for debugging only)
# message("CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
# message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")

# Required external MPI libraries 
FIND_PACKAGE(MPI REQUIRED)

# Required Boost libraries 
SET(BOOST_MIN_VERSION "1.41.0")
FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS mpi serialization program_options date_time)

INCLUDE_DIRECTORIES( ${MPI_CXX_INCLUDE_PATH} )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES(src)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(tools)

