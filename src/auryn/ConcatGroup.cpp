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

#include "ConcatGroup.h"

using namespace auryn;


ConcatGroup::ConcatGroup( ) : SpikingGroup()
{
	evolve_locally_bool = true;
}

ConcatGroup::~ConcatGroup()
{
	if ( evolve_locally() ) {
	}
}

void ConcatGroup::update_parameters()
{
	NeuronID new_size = 0;
	for (unsigned int i = 0 ; i < parents.size() ; ++i) {
		new_size += parents.at(i)->get_size();
	}
	size = new_size;
}

void ConcatGroup::add_parent_group(SpikingGroup * group)
{
	parents.push_back(group);
}


void ConcatGroup::copy_spikes_and_attribs(SpikingGroup * group, NeuronID group_offset, AttributeContainer * attrib_container)
{
}

void ConcatGroup::evolve()
{
	SpikeContainer tmp_spikes;
	AttributeContainer tmp_attribs;
	SpikeContainer tmp_spikes_imm;
	AttributeContainer tmp_attribs_imm;

	for (unsigned int i = 0 ; i < parents.size() ; ++i) {
		SpikingGroup * g = parents.at(i);
		// std::copy(g->get_spikes()->begin(), g->get_spikes()->end(), std::back_inserter(tmp_spikes));
		std::copy(g->get_attributes()->begin(), g->get_attributes()->end(), std::back_inserter(tmp_attribs));
	}

	spikes = new SpikeContainer(tmp_spikes);
	attribs = new AttributeContainer(tmp_attribs);
}



