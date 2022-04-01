/* 
* Copyright 2014-2020 Friedemann Zenke
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

#ifndef CONCATGROUP_H_
#define CONCATGROUP_H_

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
 * the seed function to seed it randomly if needed. Note that all ConcatGroups
 * in a simulation share the same random number generator. Therefore it 
 * sufficed to seed one of them.
 */
class ConcatGroup : public SpikingGroup
{
private:
	std::vector<SpikingGroup*> parents;


	void init(AurynDouble rate);
	void update_parameters();

protected:
	
public:
	/*! Standard constructor. 
	 */
	ConcatGroup( );
	/*! Default destructor */
	virtual ~ConcatGroup();

	/*! Add parent group */
	copy_spikes_and_attribs(SpikingGroup * group, NeuronID group_offset, AttributeContainer * attrib_container);

	/*! Add parent group */
	void  add_parent_group(SpikingGroup * group);

	/*! Evolve function for internal use by System */
	virtual void evolve();
};

}

#endif /*CONCATGROUP_H_*/
