/* 
* Copyright 2014-2016 Friedemann Zenke
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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#ifndef POISSONSTIMULATOR_H_
#define POISSONSTIMULATOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "Logger.h"
#include "Monitor.h"
#include "NeuronGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/poisson_distribution.hpp>

namespace auryn {

/*! \brief Stimulator class to inject timeseries of currents NeuronGroups
 * 
 *  Instances of this class inject independent currents from a Poisson distribution to the NeuronGroup assigned.
 */

class PoissonStimulator : protected Monitor
{
private:

	static boost::mt19937 gen; 
	boost::poisson_distribution<int> * dist;
	boost::variate_generator<boost::mt19937&, boost::poisson_distribution<int> > * die;


	/*! Vector storing all the current values */
	AurynState * currents;

	/*! Vector storing all new current values */
	AurynState * newcurrents;

	/*! Target membrane */
	auryn_vector_float * target_vector;

	/*! Scale stimulus rate */
	AurynFloat poisson_rate;

	/*! Scale stimulus size */
	AurynFloat poisson_weight;

	/*! Default init method */
	void init(NeuronGroup * target, AurynFloat rate, AurynWeight w );

	void free();

	/*! Returns the lambda parameter of the pmf for Poisson. */
	AurynFloat get_lambda();

protected:

	/*! The target NeuronGroup */
	NeuronGroup * dst;

	
public:
	/*! \brief Default Constructor 
	 * @param[target] The target spiking group. 
	 * @param[rate]   The firing rate of each the Poisson process.
	 * @param[weight] The weight or unit of amount of change on the state variable
	 */
	PoissonStimulator(NeuronGroup * target, AurynFloat rate=100.0, AurynWeight w = 0.1 );

	/*! \brief Default Destructor */
	virtual ~PoissonStimulator();


	/*! \brief Sets the event rate of the underlying Poisson generator 
	 *
	 * @param[rate] The Poisson rate */
	void set_rate(AurynFloat rate);

	/*! \brief Returns the event rate of the underlying Poisson generator. */
	AurynFloat get_rate();

	/*! \brief Seeds the random number generator of all PoissonStimulator objects on this rank
	 *
	 * @param[s] The random seed. 
	 * Note, that this seeding function is not rank save. To ensure that the currents are 
	 * independent on different ranks you need to give a different seed on each rank when
	 * running parallel simulations. */
	void seed(int s);


	/*! \brief Sets the state that is stimulated 
	 *
	 * This must be a valid state vector name (default = mem). */
	void set_target_state( string state_name = "mem" );

	/*! \brief Implementation of necessary propagate() function. */
	void propagate();

};

}

#endif /*POISSONSTIMULATOR_H_*/
