#define BOOST_TEST_MODULE EulerTrace test 
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "auryn.h"

using namespace auryn;

BOOST_AUTO_TEST_CASE( EulerTrace_evolve ) {

	double tau = 20e-3; // a typical STDP time constant
	EulerTrace * tr_euler = new EulerTrace(4,tau);
	tr_euler->get_state_ptr()->set_all(1.0);
	// LinearTrace * tr_linear = new LinearTrace(2,tau,sys->get_clock_ptr());

	float simtime = 0.1;
	float maxdev = 0.0;

	for ( int i = 0 ; i < (int)(simtime/auryn::auryn_timestep) ; ++i ) {
		float solution = std::exp(-i*auryn::auryn_timestep/tau);
		float deviation = std::abs(tr_euler->get(0)-solution)/solution;
		if ( maxdev < deviation ) maxdev = deviation;
		tr_euler->evolve();
	}

	float precision = 1e-3;
	BOOST_REQUIRE( maxdev < precision );
}

BOOST_AUTO_TEST_CASE( EulerTrace_follow ) {
	double tau = 20e-3; // a typical STDP time constant
	EulerTrace * tr_euler = new EulerTrace(4,tau);

	AurynStateVector * target = new AurynStateVector(4);
	target->set_all(3.141592);

	tr_euler->set_target( target );

	// set_target inits with copy
	BOOST_REQUIRE( target->get(0) == target->get(0) );

	// lets drive the trace away
	tr_euler->set(0,0.0);
	BOOST_REQUIRE( tr_euler->get(0) == 0.0 );

	// The follow trace should go exponentially to the target value 3.14..
	AurynStateVector * dist = new AurynStateVector(4);
	for ( AurynTime t = 0 ; t < 6*tau/auryn_timestep; ++t ) {
		tr_euler->follow();
		dist->diff(target, tr_euler);
		// std::cout << tr_euler->get(0) << " " << dist->get(0) << std::endl;
	}

	// and be almost there after 6 tau 
	float precision = 1e-4;
	float result = 3.141592*std::exp(-6.0) + precision;
	BOOST_REQUIRE( dist->get(0) < result );
}
