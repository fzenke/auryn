#define BOOST_TEST_MODULE LinearTrace test
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "auryn.h"

using namespace auryn;

const NeuronID idx = 1;
const double tau = 20e-3;
const float precision = 1e-3;

/*! \brief Test decay from 1 of LinearTrace */
BOOST_AUTO_TEST_CASE( LinearTrace_decay ) {
	// LinearTrace needs a clock to compute time intervals
	AurynTime clk = 0;
	LinearTrace * tr_linear = new LinearTrace(4,tau,&clk);

	BOOST_CHECK_EQUAL( tr_linear->get(idx), 0.0 );

	tr_linear->inc(idx);
	BOOST_CHECK_EQUAL( tr_linear->get(idx), 1.0 );


	for ( int i = 0 ; i < 10 ; ++i ) {
		clk = i*123;
		float solution  = std::exp(-auryn::auryn_timestep*clk/tau);
		float deviation = std::abs(tr_linear->get(idx)-solution)/solution;
		// std::cout << solution << " " << tr_linear->get(idx) << std::endl;
		BOOST_CHECK( deviation < precision );
	}
}

/*! \brief Test decay from 1 of LinearTrace for ver large tau=500s 
 *
 * EulerTrace fails to get this right. */
BOOST_AUTO_TEST_CASE( LinearTrace_slow_decay ) {
	// EulerTrace typically failes for tau > 200s
	double tau_long = 500;

	// LinearTrace needs a clock to compute time intervals
	AurynTime clk = 0;
	LinearTrace * tr_linear = new LinearTrace(4,tau_long,&clk);

	BOOST_CHECK_EQUAL( tr_linear->get(idx), 0.0 );

	tr_linear->inc(idx);
	BOOST_CHECK_EQUAL( tr_linear->get(idx), 1.0 );


	for ( int i = 0 ; i < 10 ; ++i ) {
		clk = i*123.0/auryn::auryn_timestep;
		float solution  = std::exp(-auryn::auryn_timestep*clk/tau_long);
		float deviation = std::abs(tr_linear->get(idx)-solution)/solution;
		// std::cout << solution << " " << tr_linear->get(idx) << std::endl;
		BOOST_CHECK( deviation < precision );
	}
}

/*! \brief Test multiple 'spikes' and compare to EulerTrace */
BOOST_AUTO_TEST_CASE( LinearTrace_pileup ) {

	// LinearTrace needs a clock to compute time intervals
	AurynTime clk = 0;


	EulerTrace * tr_euler = new EulerTrace(4,tau); // we use an EulerTrace as reference
	LinearTrace * tr_linear = new LinearTrace(4,tau,&clk);

	float maxdev = 0.0;
	int simsteps = 1.0e-3/auryn::auryn_timestep;
	for ( int k = 0 ; k < 10 ; ++ k ) {
		tr_euler->inc(idx);
		tr_linear->inc(idx);

		// evolve EulerTrace for some time steps
		for ( int i = 0 ; i < simsteps ; ++i ) {
			tr_euler->evolve();
			// std::cout << tr_euler->get(idx) << std::endl;
			clk++;
		}

		// compare
		const float solution  = tr_euler->get(idx);
		const float deviation = std::abs(tr_linear->get(idx)-solution)/solution;
		if ( maxdev < deviation ) maxdev = deviation;
		// std::cout << solution << " " << tr_linear->get(idx) << std::endl;

		// evolve EulerTrace for some time steps
		for ( int i = 0 ; i < k*simsteps ; ++i ) {
			tr_euler->evolve();
			clk++;
		}

	}

	BOOST_REQUIRE( maxdev < precision );
}

