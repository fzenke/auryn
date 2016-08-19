# Include all files in the src directory
FILE(GLOB auryn_SRC "*.h" "*.cpp" "auryn/*.cpp" "auryn/*.h" )
ADD_LIBRARY( auryn STATIC ${auryn_SRC} )

# message("AURYN_EXT_LINKLIBS is ${AURYN_EXT_LINKLIBS}")
# message("AURYN_EXT_INCLUDEDIRS is ${AURYN_EXT_INCLUDEDIRS}")

# Dependencies to external libraries
TARGET_LINK_LIBRARIES( auryn ${AURYN_EXT_LINKLIBS} )
TARGET_INCLUDE_DIRECTORIES( auryn PUBLIC ${AURYN_EXT_INCLUDEDIRS} )

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


