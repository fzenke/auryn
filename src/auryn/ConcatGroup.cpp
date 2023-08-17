/* 
* Copyright 2014-2023 Friedemann Zenke
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


ConcatGroup::ConcatGroup( ) : SpikingGroup( 0, ROUNDROBIN )
{
	sys->register_spiking_group(this);
	evolve_locally_bool = true;
	active = true;
	// FIXME make sure this runs under all conditions
	// mode = ROUNDROBIN;
}

ConcatGroup::~ConcatGroup()
{
	if ( evolve_locally() ) {
	}
}

void ConcatGroup::add_parent_group(SpikingGroup * group)
{
	parents.push_back(group);
	size += group->get_size();
	rank_size += group->get_post_size();
}

void ConcatGroup::copy_spikes(SpikeContainer * src, NeuronID group_offset)
{
	// std::cout << "copy_spikes" << std::endl;
	for (SpikeContainer::const_iterator spike = src->begin() ;
			spike != src->end() ; 
			++spike ) {
		spikes->push_back(*spike+group_offset);
	}
}

void ConcatGroup::copy_attributes(AttributeContainer * src)
{
	for (AttributeContainer::const_iterator attrib = src->begin() ;
			attrib != src->end() ; 
			++attrib ) {
		attribs->push_back(*attrib);
	}
}

void ConcatGroup::evolve()
{
	NeuronID offset = 0;
	for (unsigned int i = 0 ; i < parents.size() ; ++i) {
		SpikingGroup * g = parents.at(i);
		copy_spikes(g->get_spikes_immediate(), offset); 
		copy_attributes(g->get_attributes_immediate()); 
		offset += g->get_size();
	}
}



