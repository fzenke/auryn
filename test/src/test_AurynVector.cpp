#define BOOST_TEST_MODULE AurynVector test 
#include <boost/test/unit_test.hpp>
#include "auryn/AurynVector.h"

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

	v.resize( n/5 );
	BOOST_CHECK_EQUAL( v.size, n/5 );

	for ( int i = 0 ; i < n/5 ; ++i ) 
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


// EOF
