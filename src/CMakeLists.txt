# Include all files in the src directory
FILE(GLOB auryn_SRC "*.h" "*.cpp" )
ADD_LIBRARY( auryn STATIC ${auryn_SRC} )

# Dependencies to external libraries
TARGET_LINK_LIBRARIES( auryn ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES( auryn PUBLIC ${Boost_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
