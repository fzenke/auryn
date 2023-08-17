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

#include "VoltageClamp.h"

using namespace auryn;

VoltageClamp::VoltageClamp(NeuronGroup * target, std::string time_series_file, std::string neuron_state_name ) : FileCurrentInjector(target, time_series_file, neuron_state_name )
{

}



VoltageClamp::~VoltageClamp()
{

}


AurynState VoltageClamp::get_current_clamping_value()
{

	AurynTime index = sys->get_clock();
	if ( loop ) {
		index = sys->get_clock()%loop_grid;
	} 

	if ( index < current_time_series->size() ) {
		return current_time_series->at( index );
	} else {
		return 0.0;
	}
}

void VoltageClamp::execute()
{
	if ( dst->evolve_locally() ) {
		AurynState cur = get_current_clamping_value();
		switch ( mode ) {
			case LIST:
				for ( int i = 0 ; i < target_neuron_ids->size() ; ++i ) {
					target_vector->set(target_neuron_ids->at(i),cur);
				}
				break;
			case ALL:
			default:
				target_vector->set_all(cur);
		}
	}
}

