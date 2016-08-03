file( GLOB TEST_SOURCES test_*.cpp )
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK) 
foreach( sourcepath ${TEST_SOURCES} )
	get_filename_component( sourcefile ${sourcepath} NAME )
	string( REPLACE ".cpp" "" simname ${sourcefile} )
	add_executable( ${simname} ${sourcefile} )
	target_link_libraries( ${simname} auryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )
endforeach( sourcepath ${TEST_SOURCES} )

