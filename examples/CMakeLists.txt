file( GLOB SIM_SOURCES *.cpp )
foreach( sourcepath ${SIM_SOURCES} )
	get_filename_component( sourcefile ${sourcepath} NAME )
	string( REPLACE ".cpp" "" simname ${sourcefile} )
	add_executable( ${simname} ${sourcefile} )
	target_link_libraries( ${simname} auryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )
endforeach( sourcepath ${SIM_SOURCES} )

