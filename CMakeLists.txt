CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)
PROJECT(Auryn)

# The version number.
SET(Auryn_VERSION_MAJOR 0)
SET(Auryn_VERSION_MINOR 8)

# Important GCC Compiler flags for Auryn's performance
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math -pipe")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pedantic")

# Display flags (for debugging only)
# message("CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
# message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")

# Find external MPI libraries 
FIND_PACKAGE(MPI REQUIRED)

# Find Boost libraries 
SET(BOOST_MIN_VERSION "1.41.0")
IF(MPI_FOUND)
	FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS mpi serialization program_options date_time unit_test_framework system filesystem)
ELSE()
	FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS serialization program_options date_time unit_test_framework system filesystem)
ENDIF()
# TODO add align to the components list once AurynVector uses it


IF(MPI_FOUND)
	set(AURYN_EXT_LINKLIBS ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES} )
	set(AURYN_EXT_INCLUDEDIRS ${MPI_CXX_INCLUDE_PATH} ${Boost_INCLUDE_DIRS} )
ELSE()
	set(AURYN_EXT_LINKLIBS  ${Boost_LIBRARIES} )
	set(AURYN_EXT_INCLUDEDIRS  ${Boost_INCLUDE_DIRS} )
ENDIF()


# INCLUDE_DIRECTORIES( ${AURYN_EXT_INCLUDEDIRS} )
INCLUDE_DIRECTORIES( src )

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(tools)

# Unit tests
ADD_SUBDIRECTORY(test/src)
