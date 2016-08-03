#define BOOST_TEST_MODULE ComplexMatrix test 
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "ComplexMatrix.cpp"

// using namespace auryn;

int element(int i, int j, int offset=1) { return (i*(j+offset))%257; }

BOOST_AUTO_TEST_CASE( dense_fill_and_read ) {

	int nb_pre = 123;
	int nb_post = 13;
	auryn::ComplexMatrix<int> mat(nb_pre,nb_post,nb_pre*nb_post);

	for (int j = 0 ; j<nb_pre ; j += 1)
	{
		for (int i = 0 ; i<nb_post ; i += 1) {
			mat.push_back(j,i,element(j,i));
		}
	}
	mat.fill_zeros();


	for (int j = 0 ; j<nb_pre ; j += 1)
	{
		for (int i = 0 ; i<nb_pre ; i += 1) {
			int el = mat.get(j,i);
			if ( el ) {
				BOOST_CHECK_EQUAL( element(j,i), el );
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( sparse_fill_and_read ) {

	int nb_pre = 12;
	int nb_post = 17;
	auryn::ComplexMatrix<int> mat(nb_pre,nb_post,nb_pre*nb_post);

	int count = 0;
	for (int j = 0 ; j<nb_pre ; j += 1)
	{
		for (int i = 0 ; i<nb_post; i += 1) {
			if ( element(j,i)%2 ) {
				mat.push_back(j,i,element(j,i));
				count++;
			}
		}
	}
	mat.fill_zeros();
	BOOST_CHECK_EQUAL( mat.get_nonzero(), count );

	for (int j = 0 ; j<nb_pre ; j += 1)
	{
		for (int i = 0 ; i<nb_post ; i += 1) {
			int el = mat.get(j,i);
			if ( element(j,i)%2 ) {
				BOOST_CHECK_EQUAL( el, element(j,i) );
			} else {
				std::cout << j << " " << i << " sh:" << element(j,i) << " is:" << el << std::endl;
				BOOST_CHECK_EQUAL( el, 0 );
			}
		}
	}
}

// BOOST_AUTO_TEST_CASE( sparse_complex_fill_and_read ) {
// 
// 	int nb_pre = 65;
// 	int nb_post = 17;
// 	auryn::ComplexMatrix<int> mat(nb_pre,nb_post,nb_pre*nb_post);
// 	mat.set_num_synaptic_states(3);
// 
// 	// sparse fill
// 	for (int j = 0 ; j<nb_pre ; j += 1)
// 	{
// 		for (int i = 0 ; i<nb_post; i += 1) {
// 			if ( element(j,i)%2 ) {
// 				mat.push_back(j,i,element(j,i));
// 			}
// 		}
// 	}
// 	mat.fill_zeros();
// 
// 	auryn::AurynVector<int, auryn::AurynLong> * sv = mat.get_synaptic_state_vector(2);
// 	sv->set_random_normal(0.0,20);
// 	auryn::AurynVector<int, auryn::AurynLong> * cp = new auryn::AurynVector<int, auryn::AurynLong>(mat.get_synaptic_state_vector(2));
// 	mat.get_synaptic_state_vector(1)->diff(mat.get_synaptic_state_vector(0),mat.get_synaptic_state_vector(2));
// 
// 	for (int j = 0 ; j<nb_pre ; j += 1)
// 	{
// 		for (int i = 0 ; i<nb_pre ; i += 1) {
// 			int el = mat.get(j,i);
// 			if ( el%2 ) {
// 				BOOST_CHECK_EQUAL( el, element(j,i) );
// 				// BOOST_CHECK_EQUAL( mat.get(i,j), element(j,i) );
// 			}
// 		}
// 	}
// }
