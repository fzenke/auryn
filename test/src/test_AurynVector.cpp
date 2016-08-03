#define BOOST_TEST_MODULE AurynVector test 
#include <boost/test/unit_test.hpp>
#include "../../src/AurynVector.h"

BOOST_AUTO_TEST_CASE( resizing ) {

	int n = 10;
    auryn::AurynVector<int> v( n );
	for ( int i = 0 ; i < n ; ++i ) v.set(i,i);

    BOOST_CHECK_EQUAL( v.size, n );

	for ( int i = 0 ; i < n ; ++i ) 
		BOOST_CHECK_EQUAL( v.data[i], i );

	v.resize( 2*n );
	BOOST_CHECK_EQUAL( v.size, 2*n );

	for ( int i = 0 ; i < n ; ++i ) 
		BOOST_CHECK_EQUAL( v.data[i], i );

	for ( int i = n ; i < 2*n ; ++i ) 
		BOOST_CHECK_EQUAL( v.data[i], 0 );

	v.resize( n/2 );
	BOOST_CHECK_EQUAL( v.size, n/2 );

	for ( int i = 0 ; i < n/2 ; ++i ) 
		BOOST_CHECK_EQUAL( v.data[i] , i );
}

BOOST_AUTO_TEST_CASE( pow2 ) {
	int n = 10;
    auryn::AurynVector<float> v( n );
	for ( int i = 0 ; i < n ; ++i ) v.set(i,i);

	v.pow(2);
	for ( int i = 0 ; i < n ; ++i ) 
		BOOST_CHECK_EQUAL( v.data[i] , (float)i*(float)i );
}

BOOST_AUTO_TEST_CASE( fast_exp ) {
	// checks for less than one percent relative
	// deviation on the interval -2.0..1.6
	float max = 2.0;
	float tolerance = 1e-2;
	int n = 10;

    auryn::AurynVector<float> v( n );
	for ( int i = 0 ; i < n ; ++i ) {
		v.set(i,1.0*(i-n/2)/(n/2)*max);
	}

	v.fast_exp();
	for ( int i = 0 ; i < n ; ++i ) {
		float answer = std::exp(1.0*(i-n/2)/(n/2)*max);
		float deviation = std::abs(v.data[i]-answer)/answer;
		// std::cout << deviation << std::endl;
		BOOST_CHECK( deviation < tolerance );
	}
}

// EOF
