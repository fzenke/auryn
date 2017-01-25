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

#ifndef POISSONGROUP_H_
#define POISSONGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "SpikingGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

namespace auryn {

/*! \brief A SpikingGroup that creates poissonian spikes with a given rate.
 *
 * This is the standard Poisson spike generator of Auryn. It implements a 
 * group of given size of Poisson neurons all firing at the same rate. 
 * The implementation is very efficient if the rate is constant throughout.
 *
 * The random number generator will be seeded identically every time. Use 
 * the seed function to seed it randomly if needed. Note that all PoissonGroups
 * in a simulation share the same random number generator. Therefore it 
 * sufficed to seed one of them.
 */
class PoissonGroup : public SpikingGroup
{
private:
	AurynTime * clk;
	AurynDouble lambda;
	static boost::mt19937 gen; 
	boost::exponential_distribution<> * dist;
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > * die;

	unsigned int salt;

	void init(AurynDouble rate);

protected:
	NeuronID x;
	
public:
	/*! Standard constructor. 
	 * @param n is the size of the SpikingGroup, i.e. the number of Poisson neurons.
	 * @param rate is the mean firing rate of all poisson neurons in the group.
	 */
	PoissonGroup(NeuronID n, AurynDouble rate=5. );
	/*! Default destructor */
	virtual ~PoissonGroup();
	/*! Evolve function for internal use by System */
	virtual void evolve();
	/*! Setter for the firing rate of all neurons. This can be used to change
	 * the firing rate during the simulation. Note that changes might have a short
	 * latency due to the internal workings of the simulator. Try avoid setting 
	 * the firing rate in every other timestep because it will reduce performance.
	 */
	void set_rate(AurynDouble rate);
	/*! Standard getter for the firing rate variable. */
	AurynDouble get_rate();
	/*! Use this to seed the random number generator. */
	void seed(unsigned int s);
};

}

#endif /*NEURONGROUP_H_*/
