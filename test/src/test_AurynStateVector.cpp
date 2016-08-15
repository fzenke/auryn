#define BOOST_TEST_MODULE AurynStateVector test 
#include <boost/test/unit_test.hpp>
#include "auryn/AurynVector.h"
#include "auryn/auryn_definitions.h"

int expected_size( int n ) 
{
	const int n_simd = SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
	int n_expected = n;
	if ( n%n_simd ) n_expected = ((n/n_simd)+1)*n_simd;
	return n_expected;
}

BOOST_AUTO_TEST_CASE( resizing ) {

	int n = 10;

    auryn::AurynStateVector v( n );
	for ( int i = 0 ; i < n ; ++i ) v.set(i,i);

    BOOST_CHECK_EQUAL( v.size, expected_size(n) );

	for ( int i = 0 ; i < n ; ++i ) 
		BOOST_CHECK_EQUAL( v.data[i], i );

	v.resize( 2*n );
	BOOST_CHECK_EQUAL( v.size, expected_size(2*n) );

	for ( int i = 0 ; i < n ; ++i ) 
		BOOST_CHECK_EQUAL( v.data[i], i );

	for ( int i = n ; i < 2*n ; ++i ) 
		BOOST_CHECK_EQUAL( v.data[i], 0 );

	v.resize( n/5 );
	BOOST_CHECK_EQUAL( v.size, expected_size(n/5) );

	for ( int i = 0 ; i < n/5 ; ++i ) 
		BOOST_CHECK_EQUAL( v.data[i] , i );
}

BOOST_AUTO_TEST_CASE( pow2 ) {
	int n = 10;
    auryn::AurynStateVector v( n );
	for ( int i = 0 ; i < n ; ++i ) v.set(i,i);

	v.pow(2);
	for ( int i = 0 ; i < n ; ++i ) 
		BOOST_CHECK_EQUAL( v.data[i] , (float)i*(float)i );
}

BOOST_AUTO_TEST_CASE( write_read ) {
	int n = 10;
    auryn::AurynStateVector v1( n );
    auryn::AurynStateVector v2( n );

	for ( int i = 0 ; i < n ; ++i ) v1.set(i,i);
	for ( int i = 0 ; i < n ; ++i ) v2.set(i,i);

	for ( int i = 0 ; i < n ; ++i ) 
		BOOST_CHECK_EQUAL( v1.get(i), i );

	for ( int i = 0 ; i < n ; ++i ) 
		BOOST_CHECK_EQUAL( v2.get(i), i );
}

BOOST_AUTO_TEST_CASE( add ) {
	int n = 10;
    auryn::AurynStateVector * v1 = new auryn::AurynStateVector( n );
    auryn::AurynStateVector * v2 = new auryn::AurynStateVector( n );

	for ( int i = 0 ; i < n ; ++i ) v1->set(i,i);
	for ( int i = 0 ; i < n ; ++i ) v2->set(i,i);

	v1->add(v2);

	for ( int i = 0 ; i < n ; ++i ) {
		float f = i;
		BOOST_CHECK_EQUAL( v1->get(i), f+f );
	}
}


// EOF
