FILE( GLOB TEST_SOURCES test_*.cpp )
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK) 
FOREACH( sourcepath ${TEST_SOURCES} )
	GET_FILENAME_COMPONENT( sourcefile ${sourcepath} NAME )
	STRING( REPLACE ".cpp" "" simname ${sourcefile} )
	ADD_EXECUTABLE( ${simname} ${sourcefile} )
	TARGET_LINK_LIBRARIES( ${simname} auryn ${AURYN_EXT_LINKLIBS} )
ENDFOREACH( sourcepath ${TEST_SOURCES} )

