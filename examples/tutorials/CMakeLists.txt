FILE( GLOB SIM_SOURCES *.cpp )
FOREACH( sourcepath ${SIM_SOURCES} )
	GET_FILENAME_COMPONENT( sourcefile ${sourcepath} NAME )
	STRING( REPLACE ".cpp" "" simname ${sourcefile} )
	ADD_EXECUTABLE( ${simname} ${sourcefile} )
	TARGET_LINK_LIBRARIES( ${simname} auryn ${AURYN_EXT_LINKLIBS} )
ENDFOREACH( sourcepath ${SIM_SOURCES} )

