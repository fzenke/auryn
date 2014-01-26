/* 
* Copyright 2014 Friedemann Zenke
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
*/

#include "StateMonitor.h"

StateMonitor::StateMonitor(NeuronGroup * source, NeuronID id, string statename, string filename, AurynTime stepsize)  
{
	init(source,id,statename,filename,stepsize);
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
	if ( source->evolve_locally() )
		target_variable = src->get_state_vector(statename)->data+nid;
	outfile << setiosflags(ios::fixed) << setprecision(6);
}

void StateMonitor::propagate()
{
	if ((sys->get_clock())%ssize==0 && src->get_rank_size() > nid ) {
		outfile << dt*(sys->get_clock()) << " " << *target_variable << "\n";
	}
}
