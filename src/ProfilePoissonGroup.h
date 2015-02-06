/* 
* Copyright 2014-2015 Friedemann Zenke
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
#include "System.h"
#include "SpikingGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

#define PROFILEPOISSON_LOAD_MULTIPLIER 0.01

using namespace std;

/*! \brief A SpikingGroup that creates poissonian spikes with a given rate
 * and spacial profile.
 */
class ProfilePoissonGroup : public SpikingGroup
{
private:
	AurynTime * clk;
	AurynDouble lambda;

	static boost::mt19937 gen; 
	boost::uniform_01<> * dist;
	boost::variate_generator<boost::mt19937&, boost::uniform_01<> > * die;

	void init(AurynDouble rate);

protected:
	NeuronID x;
	AurynDouble jumpsize;

	AurynFloat * profile; //!< stores the spatial distribution of relative firing rates
	
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
	void set_profile(AurynFloat * newprofile);
	void set_flat_profile();
	void set_gaussian_profile(AurynDouble  mean, AurynDouble sigma, AurynDouble floor=0.0);

	/*! Standard getter for the firing rate variable. */
	AurynDouble get_rate();

	/*! Use this to seed the random number generator. */
	void seed(int s);
};

#endif /*NEURONGROUP_H_*/
