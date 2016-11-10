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

#ifndef MOVINGBUMPGROUP_H_
#define MOVINGBUMPGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
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
 * This type of stimulation can be used as a standart sanity test for plastic Connection classes to 
 * see whether or not they develop selectivity through some sort of competitive mechanism.
 *
 */
class MovingBumpGroup : public ProfilePoissonGroup
{
private:
	static boost::mt19937 order_gen;

	/*! \brief Stimulus duration after which the bump switches to a new random location. */
	AurynTime stimulus_duration;

	/*! \brief Inter ctimulus interval in s. */
	AurynTime stimulus_interval;

	/*! \brief Width of the Gaussian in number of neurons */
	NeuronID profile_width;

	/*! \brief Floor firing rate relative to max */
	AurynFloat floor_;

	/*! \brief File stream handle for output file of stimulation time series */
	std::ofstream tiserfile;

	/*! \brief Next event time */
	AurynTime next_event;

	/*! \brief True when stimulus bump is on */
	bool stimulus_active;


	void init ( AurynFloat duration, AurynFloat width, string outputfile );
	
public:

	/*! \brief Minimal relative position in group for center of bump (default = 0) */
	double pos_min;
	/*! \brief Maximum relative position in group for center of bump (default = 1) */
	double pos_max;

	/*! \brief Default constructor 
	 * \param n Size of the group
	 * \param duration Duration of constant stimulation with a given stimulus profile in s
	 * \param width The relative width of the bump
	 * \param rate Base firing rate
	 * \param tiserfile Timeseries file for logging of bump position
	 */
	MovingBumpGroup(NeuronID n, 
			AurynFloat duration, 
			AurynFloat width, 
			AurynDouble rate=5.0,
			string tiserfile = "stimulus.dat" );

	/*! \brief Sets firing rate floor 
	 *
	 * Floor is given in relative units with respect to the maximum amplitude in the Gaussian
	 */
	void set_floor(AurynFloat floor);

	/*! \brief Sets width of Gaussian rate profile 
	 *
	 * Width is given in relative units of neurons and characterizes the stdev of the Gaussian dist
	 */
	void set_width(NeuronID width);

	/*! \brief Sets stimulus duration
	 *
	 * duration is given in units of s
	 *
	 * \param duration the new stimulus duration in s 
	 */
	void set_duration(AurynFloat duration);

	/*! \brief Sets inter-stimulus interval
	 *
	 * \param interval given in units of s
	 */
	void set_interval(AurynFloat interval);

	virtual ~MovingBumpGroup();
	virtual void evolve();
};

}

#endif /*MOVINGBUMPGROUP_H_*/
