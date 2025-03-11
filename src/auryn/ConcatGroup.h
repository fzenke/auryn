/* 
* Copyright 2014-2025 Friedemann Zenke
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

/*! \brief ConcatGroup is a meta SpikingGroup which concatenates neurons from several parent groups.
 *
 * This is an EXPERIMENTAL group which has not undergone extensive testing yet.
 *
 * When initialized the group does not have any own neurons.
 * Add parent groups using the add_parent_group function.
 * The group will emit spikes from all the neurons in the parent groups in a concatenated way.
 */
class ConcatGroup : public SpikingGroup
{
private:
	std::vector<SpikingGroup*> parents;


	void init(AurynDouble rate);
	void update_parameters();

protected:
	/*! \brief Copy spikes from container to local delay */
	void copy_spikes(SpikeContainer * src, NeuronID group_offset);

	/*! \brief Copy spike attributes from group container to local delay */
	void copy_attributes(AttributeContainer * group);
	
public:
	/*! Standard constructor. 
	 */
	ConcatGroup( );

	/*! Default destructor */
	virtual ~ConcatGroup();

	/*! Add parent group */
	void  add_parent_group(SpikingGroup * group);

	/*! Evolve function for internal use by System */
	virtual void evolve();
};

}

#endif /*CONCATGROUP_H_*/
