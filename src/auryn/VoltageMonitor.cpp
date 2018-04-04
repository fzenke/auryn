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

#include "VoltageMonitor.h"

using namespace auryn;

VoltageMonitor::VoltageMonitor(NeuronGroup * source, NeuronID id, std::string filename, AurynDouble stepsize) : StateMonitor(source, id, "mem", filename, stepsize)
{
	init();
}

VoltageMonitor::~VoltageMonitor()
{
}

void VoltageMonitor::init()
{
	paste_spikes = true;
	target_variable = src->get_state_vector("mem")->ptr(nid);
}

void VoltageMonitor::execute()
{
	if ( auryn::sys->get_clock() < t_stop ) {
		// we output spikes irrespectively of the sampling interval, because 
		// the membrane potential isn't a smooth function for most IF models when
		// they spike, so it's easy to "miss" a spike otherwise
		double voltage = *target_variable;
		if ( paste_spikes ) {
			SpikeContainer * spikes = src->get_spikes_immediate();
			for ( int i = 0 ; i < spikes->size() ; ++i ) {
				if ( spikes->at(i) == nid ) {
					voltage = VOLTAGEMONITOR_PASTED_SPIKE_HEIGHT;
					outfile << (auryn::sys->get_time()) << " " << voltage << "\n";
					return;
				}
			}
		}
		if ( (auryn::sys->get_clock())%ssize==0 )
			outfile << (auryn::sys->get_time()) << " " << voltage << "\n";
	}
}


