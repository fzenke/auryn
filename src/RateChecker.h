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

#ifndef RATECHECKER_H_
#define RATECHECKER_H_

#include "auryn_definitions.h"
#include "System.h"
#include "Checker.h"
#include "SpikingGroup.h"

namespace auryn {

/*! \brief A Checker class that tracks population firing rate as a moving
 * average and breaks a run if it goes out of bound.
 *
 * This class should be specified at least once with all plastic runs to ensure
 * that expoding firing rates or a silent network does not get simulated
 * needlessly for hours.
 *
 * The different constructors allow to specify different min and max firing
 * rates to guard against too active or quiet networks.  Also the timeconstant
 * (tau) over which the moving rate average is computed online can be specified.
 * Allow for 3-5 x tau for the estimate to settle to its steady state value.  To
 * avoid accidental breaking of a run due to this effect, at initialization the
 * rate estimate is assumed to  be the mean of the min and max. Note further
 * that this checker computes population averages over the fraction of a neuron
 * group which is simulated on a particular rank.  In highly parallel
 * simulations when the number of neurons per rank is very the rate estimate
 * might have a high variance accross ranks.  If highly parallel simulation is
 * anticipated tau should be chosen longer to avoid spurious breaks caused by a
 * noisy rate estimate or a different checker which computes the rate of entire
 * population (after a MINDELAY s minimal delay) should be used.
 */

class RateChecker : public Checker
{
private:
	AurynFloat decay_multiplier;
	AurynDouble popmin;
	AurynDouble popmax;
	AurynDouble state;
	AurynFloat timeconstant;
    NeuronID size;
	void init(AurynFloat min, AurynFloat max, AurynFloat tau);
	
	virtual void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version );
	virtual void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version );
protected:
	SpikingGroup * src;

public:
	/*! A more elaborate constructor specifying also a minimum rate to guard against silent networks.
	 * @param source the source group to monitor.
	 * @param min the minimum firing rate below which the Checker signals a break of the simulation.
	 * @param max the maximum firing rate above which the Checker signals a break of the simulation.
	 * @param tau the time constant over which to compute the moving average of the rate.
	 */
	RateChecker(SpikingGroup * source, AurynFloat min, AurynFloat max, AurynFloat tau=1.0);
	virtual ~RateChecker();
	/*! The propagate function required for internal use. */
	virtual bool propagate();
	/*! The query function required for internal use. */
	virtual AurynFloat get_property();
	/*! Reads out the current rate estimate. */
	AurynFloat get_rate();
	/*! Sets the current rate estimate -- for instance to provide a reasonable guess upon init.
	 * ( per default this is (max+min)/2.) */
	void set_rate(AurynFloat r);
	void reset();
};

}

#endif /*RATECHECKER_H_*/
