#define BOOST_TEST_MODULE AurynDelayVector 
#include <boost/test/unit_test.hpp>
#include "auryn/auryn_definitions.h"
#include "auryn/AurynDelayVector.h"


float linfun(float x)
{
	return 0.1*x-0.3;
}

BOOST_AUTO_TEST_CASE( memory ) {

	int n = 4;
	int d = 5;

    auryn::AurynDelayVector v( n, d );

	for ( int t = 0 ; t < d ; ++t ) {
		// std::cout << "t " << t << std::endl;
		for ( int i = 0 ; i < n ; ++i ) v.set(i,linfun(i*t));
		//v.print();
		v.advance();
	}

	// check delayed by 2
	int dly = 3;
	// std::cout << "checking delayed by " << dly << std::endl;
	// v.mem_ptr(dly)->print();
	for ( int i = 0 ; i < n ; ++i ) {
		BOOST_CHECK_EQUAL( v.mem_ptr(dly)->get(i), linfun(i*(d-dly)) );
	}

	// check delayed by max delay (should be first element)
	dly = -1; // the same as dly = d
	// std::cout << "checking delayed by max " << std::endl;
	// v.mem_ptr(dly)->print();
	for ( int i = 0 ; i < n ; ++i ) {
		BOOST_CHECK_EQUAL( v.mem_ptr(dly)->get(i), linfun(i*0) );
	}
}



// EOF
