/* 
* Copyright 2014-2018 Friedemann Zenke
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

#ifndef PARROTGROUP_H_
#define PARROTGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "SpikingGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

namespace auryn {

/*! \brief A SpikingGroup that copies the output of another source SpikingGroup
 *
 * This group implements a Parrot group, a useful concept from the NEST simulator. 
 * A parrot group repeats the Spikes of a given source SpikingGroup plus an axonal 
 * transmission delay.
 */
class ParrotGroup : public SpikingGroup
{
private:
	SpikingGroup * src;

protected:
	
public:
	/*! \brief The standard constructor. 
	 * \param source
	 */
	ParrotGroup( SpikingGroup * source );

	/*! \brief Default destructor */
	virtual ~ParrotGroup();

	/*! \brief The evolve function for internal use by System class */
	virtual void evolve();
};

}

#endif /*PARROTGROUP_H_*/
