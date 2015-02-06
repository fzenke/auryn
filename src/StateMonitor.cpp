/* 
* Copyright 2014-2015 Friedemann Zenke
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

#include "StateMonitor.h"

StateMonitor::StateMonitor(NeuronGroup * source, NeuronID id, string statename, string filename, AurynDouble sampling_interval)  
{
	init(source,id,statename,filename,sampling_interval/dt);
}

StateMonitor::~StateMonitor()
{
}

void StateMonitor::init(NeuronGroup * source, NeuronID id, string statename, string filename, AurynTime stepsize)
{
	if ( !source->localrank(id) ) return; // do not register if neuron is not on the local rank

	Monitor::init(filename);
	sys->register_monitor(this);
	src = source;
	nid = source->global2rank(id);

	ssize = stepsize;
	if ( ssize < 1 ) ssize = 1;

	if ( source->evolve_locally() ) {
		target_variable = src->get_state_vector(statename)->data+nid;
	} else {
		nid = src->get_rank_size() + 1;
	}
	outfile << setiosflags(ios::fixed) << setprecision(6);
}

void StateMonitor::propagate()
{
	if ((sys->get_clock())%ssize==0 && src->get_rank_size() > nid ) {
		outfile << dt*(sys->get_clock()) << " " << *target_variable << "\n";
	}
}
