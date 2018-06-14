#define BOOST_TEST_MODULE SimpleMatrix test 
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "auryn/SimpleMatrix.h"

// using namespace auryn;

int element(int i, int j, int offset=1) { return (i*(j+offset))%257; }

BOOST_AUTO_TEST_CASE( dense_fill_and_read ) {

	int nb_pre = 11;
	int nb_post = 13;
	auryn::SimpleMatrix<int> mat(nb_pre,nb_post,nb_pre*nb_post);

	for (int j = 0 ; j<nb_pre ; j += 1)
	{
		for (int i = 0 ; i<nb_post ; i += 1) {
			mat.push_back(j,i,element(j,i));
		}
	}

	mat.fill_na();


	for (int j = 0 ; j<nb_pre ; j += 1)
	{
		for (int i = 0 ; i<nb_post ; i += 1) {
			int el = mat.get(j,i);
			if ( el ) {
				BOOST_CHECK_EQUAL( element(j,i), el );
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( sparse_fill_and_read ) {

	int nb_pre = 7;
	int nb_post = 5;
	auryn::SimpleMatrix<int> mat(nb_pre,nb_post,nb_pre*nb_post);

	int count = 0;
	for (int j = 0 ; j<nb_pre ; j += 1)
	{
		for (int i = 0 ; i<nb_post; i += 1) {
			if ( element(j,i)%2 ) {
				std::cout << "Writing i=" << i << " j=" << j << std::endl;
				mat.push_back(j,i,element(j,i));
				count++;
			}
		}
	}
	mat.fill_na();
	BOOST_CHECK_EQUAL( mat.get_nonzero(), count );

	for (int j = 0 ; j<nb_pre ; j += 1) {
		for (int i = 0 ; i<nb_post ; i += 1) {
			// std::cout << j << " " << i << " sh:" << element(j,i) << " is:" << el << std::endl;
			if ( element(j,i)%2 ) {
				// only read elements which exist (otherwise we get an error)
				std::cout << "Reading i=" << i << " j=" << j << std::endl;
				int el = mat.get(j,i);
				BOOST_CHECK_EQUAL( el, element(j,i) );
			} 
		}
	}
}

BOOST_AUTO_TEST_CASE( buffer_resize ) {

	int nb_pre = 7;
	int nb_post = 5;

	// here we reserve too much memory
	auryn::SimpleMatrix<int> mat(nb_pre,nb_post,nb_pre*nb_post*10); 

	int count = 0;
	for (int j = 0 ; j<nb_pre ; j += 1)
	{
		for (int i = 0 ; i<nb_post; i += 1) {
			if ( element(j,i)%5 ) { // push only every 5th element
				mat.push_back(j,i,element(j,i));
				count++;
			}
		}
	}
	mat.fill_na();
	BOOST_CHECK_EQUAL( mat.get_nonzero(), count );

	// now resize to make it smaller
	// we do prune here which calls resize buffer to the exact number of nonzero elements
	// std::cout << "Pruning buffer" << std::endl;
	mat.prune();

	BOOST_CHECK_EQUAL( mat.get_nonzero(), count );

	std::cout << "Verify elements" << std::endl;
	for (int j = 0 ; j<nb_pre ; j += 1) {
		for (int i = 0 ; i<nb_post ; i += 1) {
			if ( element(j,i)%5 ) {
				int el = mat.get(j,i);
				BOOST_CHECK_EQUAL( el, element(j,i) );
			} 
		}
	}

	// now resize to make the buffer bigger again 
	// std::cout << "Increasing buffer size" << std::endl;
	mat.resize_buffer(1234);

	BOOST_CHECK_EQUAL( mat.get_nonzero(), count );

	// std::cout << "Verify elements" << std::endl;
	for (int j = 0 ; j<nb_pre ; j += 1) {
		for (int i = 0 ; i<nb_post ; i += 1) {
			if ( element(j,i)%5 ) {
				int el = mat.get(j,i);
				BOOST_CHECK_EQUAL( el, element(j,i) );
			} 
		}
	}
}

