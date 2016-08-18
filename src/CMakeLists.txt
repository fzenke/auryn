# Include all files in the src directory
FILE(GLOB auryn_SRC "*.h" "*.cpp" "auryn/*.cpp" "auryn/*.h" )
ADD_LIBRARY( auryn STATIC ${auryn_SRC} )

# Dependencies to external libraries
TARGET_LINK_LIBRARIES( auryn ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES( auryn PUBLIC ${Boost_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})

INSTALL(TARGETS auryn 
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib)
INSTALL(FILES auryn.h DESTINATION include)
INSTALL(DIRECTORY auryn DESTINATION include FILES_MATCHING PATTERN "*.h")

# Uncomment the following lines to also compile a dynamic library
# ADD_LIBRARY( auryn_shared SHARED ${auryn_SRC} )
# SET_TARGET_PROPERTIES(auryn_shared PROPERTIES OUTPUT_NAME auryn)
# TARGET_LINK_LIBRARIES( auryn_shared ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})
# TARGET_INCLUDE_DIRECTORIES( auryn_shared PUBLIC ${Boost_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
# INSTALL(TARGETS auryn_shared DESTINATION lib)


