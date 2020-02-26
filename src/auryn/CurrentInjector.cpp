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

#include "CurrentInjector.h"

using namespace auryn;

CurrentInjector::CurrentInjector(NeuronGroup * target, std::string neuron_state_name, AurynFloat initial_current ) : Device( )
{
	auryn::sys->register_device(this);
	dst = target;

	set_target_state(neuron_state_name);
	currents = new AurynVectorFloat(dst->get_vector_size()); 

	currents->set_all( initial_current );
	set_scale(1.0);
}



void CurrentInjector::free( ) 
{
	delete currents;
}


CurrentInjector::~CurrentInjector()
{
	free();
}

void CurrentInjector::execute()
{
	if ( dst->evolve_locally() ) {
		target_vector->saxpy(alpha, currents);
	}
}

void CurrentInjector::set_current(NeuronID i, AurynFloat current) {
	currents->set(i, current);
}

void CurrentInjector::set_all_currents(AurynFloat current) {
	currents->set_all(current);
}

void CurrentInjector::set_target_state(std::string state_name) {
	target_vector = dst->get_state_vector(state_name);
}


void CurrentInjector::set_scale(AurynState scale) {
	alpha = scale*auryn_timestep;
}
