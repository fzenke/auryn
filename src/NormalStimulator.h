/* 
* Copyright 2014 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef NORMALSTIMULATOR_H_
#define NORMALSTIMULATOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "Logger.h"
#include "Monitor.h"
#include "NeuronGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

namespace auryn {

/*! \brief Stimulator class to inject timeseries of currents to patterns (subpopulations) of neurons 
 * 
 * Instances of this class inject currents that vary over time to subpopulations of the NeuronGroup assigned.
 */

class NormalStimulator : protected Monitor
{
private:

	static boost::mt19937 gen; 
	boost::normal_distribution<float> * dist;
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<float> > * die;


	/*! Vector storing all the current values */
	AurynState * currents;

	/*! Vector storing all new current values */
	AurynState * newcurrents;

	/*! Target membrane */
	auryn_vector_float * target_vector;

	/*! Scale stimulus size */
	AurynWeight normal_sigma;

	/*! Default init method */
	void init(NeuronGroup * target, AurynWeight sigma, string target_state);

	void free();

	/*! Returns the lambda parameter of the pmf for Normal. */
	AurynFloat get_lambda();

protected:

	/*! The target NeuronGroup */
	NeuronGroup * dst;

	
public:
	/*! Default Constructor 
	 * @param[target] The target spiking group. 
	 * @param[rate]   The firing rate of each the Normal process.
	 * @param[weight] The weight or unit of amount of change on the state variable
	 */
	NormalStimulator(NeuronGroup * target, AurynWeight sigma=1.0, string target_state="inj_current");

	/*! Default Destructor */
	virtual ~NormalStimulator();


	/*! Sets the event rate of the underlying Normal generator */
	void set_sigma(AurynFloat sigma);

	/*! Returns the  event rate of the underlying Normal generator */
	AurynFloat get_sigma();

	/*! Seeds the random number generator of all NormalStimulator objects */
	void seed(int s);


	/*! Sets the state that is stimulated with Normal input.
	 * This must be a valid state vector name (default = mem) */
	void set_target_state( string state_name = "inj_current" );

	/*! Implementation of necessary propagate() function. */
	void propagate();

};

}

#endif /*POISSONSTIMULATOR_H_*/
