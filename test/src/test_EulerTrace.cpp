#define BOOST_TEST_MODULE EulerTrace test 
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "auryn.h"

using namespace auryn;

BOOST_AUTO_TEST_CASE( EulerTrace_decay ) {

	double tau = 20e-3; // a typical STDP time constant
	EulerTrace * tr_euler = new EulerTrace(4,tau);
	tr_euler->get_state_ptr()->set_all(1.0);
	// LinearTrace * tr_linear = new LinearTrace(2,tau,sys->get_clock_ptr());

	float simtime = 0.1;
	float maxdev = 0.0;

	for ( int i = 0 ; i < (int)(simtime/dt) ; ++i ) {
		float solution = std::exp(-i*dt/tau);
		float deviation = std::abs(tr_euler->get(0)-solution)/solution;
		if ( maxdev < deviation ) maxdev = deviation;
		tr_euler->evolve();
	}

	float precision = 1e-3;
	BOOST_REQUIRE( maxdev < precision );
}

