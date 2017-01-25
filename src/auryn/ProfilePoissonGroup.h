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

#ifndef PROFILEPOISSONGROUP_H_
#define PROFILEPOISSONGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "SpikingGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

namespace auryn {

/*! \brief A SpikingGroup that creates poissonian spikes with a given rate
 * and spatial profile.
 *
 * This SpikingGroup is a logic extension of PoissonGroup for (relatively) stationary, 
 * but not uniform firing rates. It uses a similar algorithm as PoissonGroup to generate spikes
 * in which each random number yields a spike, but uses a warped output array in which neurons
 * can have different firing probabilities in each timestep. The resulting implementation requires
 * the computation of the cumulative firing probability accross the group in every timestep. It is 
 * therefore substantially slower than PoissonGroup, but presumably much faster than drawing 
 * random numbers for each neuron in each time step.
 * To set the firing rate profile use the function set_profile which needs to point to an array at 
 * which the firing rate profile is stored.
 */
class ProfilePoissonGroup : public SpikingGroup
{
private:
	AurynTime * clk;
	AurynDouble lambda;


	void init(AurynDouble rate);

protected:
	NeuronID x;
	AurynDouble jumpsize;

	static boost::mt19937 gen; 
	boost::uniform_01<> * dist;
	boost::variate_generator<boost::mt19937&, boost::uniform_01<> > * die;

	auryn_vector_float * profile; //!< stores the spatial distribution of relative firing rates
	
public:
	/*! Standard constructor. 
	 * @param n is the size of the SpikingGroup, i.e. the number of Poisson neurons.
	 * @param rate is the mean firing rate of all poisson neurons in the group.
	 */
	ProfilePoissonGroup(NeuronID n, AurynDouble rate=5. );

	/*! Default destructor */
	virtual ~ProfilePoissonGroup();

	/*! Evolve function for internal use by System */
	virtual void evolve();

	/*! Setter for the firing rate of all neurons. This can be used to change
	 * the firing rate during the simulation. Note that changes might have a short
	 * latency due to the internal workings of the simulator. Try avoid setting 
	 * the firing rate in every other timestep because it will reduce performance.
	 */
	void set_rate(AurynDouble rate);

	void normalize_profile();

	/*! \begin Sets firing rate profile to the array elements given in newprofile 
	 *
	 * Expects a vector of the size n of the SpikingGroup. */
	void set_profile(AurynFloat * newprofile);

	/*! \begin Sets firing rate profile to a state vector 
	 *
	 * Note, does not normalize the profile to a probability distribution! 
	 * Net firing rate can change! Expects a state vector with appropriate 
	 * size on the rank which is smaller than or equal to the size of this 
	 * group.*/
	void set_profile(auryn_vector_float * newprofile);

	void set_flat_profile();
	void set_gaussian_profile(AurynDouble  mean, AurynDouble sigma, AurynDouble floor=0.0);

	/*! Standard getter for the firing rate variable. */
	AurynDouble get_rate();

	/*! Use this to seed the random number generator. */
	void seed(int s);
};

}

#endif /*NEURONGROUP_H_*/
