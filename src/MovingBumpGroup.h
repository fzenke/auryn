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

#ifndef MOVINGBUMPGROUP_H_
#define MOVINGBUMPGROUP_H_

#include "auryn_definitions.h"
#include "System.h"
#include "SpikingGroup.h"
#include "ProfilePoissonGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

namespace auryn {

/*! \brief A special PoissonGroup that generates jumping Gaussian bumps in the firing rate profile
 *
 */
class MovingBumpGroup : public ProfilePoissonGroup
{
private:
	static boost::mt19937 order_gen;

	AurynTime stimulus_duration;
	AurynTime mean_isi;

	/*! Width of the Gaussian in number of neurons */
	NeuronID width;

	/*! Floor firing rate relative to max */
	AurynFloat floor;

	std::ofstream tiserfile;

	AurynTime next_event;


	void init ( AurynFloat duration, NeuronID width, string outputfile );
	
public:

	/*! \brief Default constructor 
	 * \param n Size of the group
	 * \param duration Duration of constant stimulation with a given stimulus profile in s
	 * \param width Width of stimulus in units of neurons
	 * \param rate Base firing rate
	 * \param tiserfile Timeseries file for logging of bump position
	 */
	MovingBumpGroup(NeuronID n, 
			AurynFloat duration, 
			NeuronID width, 
			AurynDouble rate=5.0,
			string tiserfile = "stimulus.dat" );

	virtual ~MovingBumpGroup();
	virtual void evolve();
};

}

#endif /*MOVINGBUMPGROUP_H_*/
