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


StateMonitor::StateMonitor(SpikingGroup * source, NeuronID id, string statename, string filename, AurynDouble sampling_interval)  
{
	init(source,id,statename,filename,sampling_interval/dt);
}

StateMonitor::StateMonitor(auryn_vector_float * state, NeuronID id, string filename, AurynDouble sampling_interval)
{
	if ( id >= state->size ) return; // do not register if neuron is out of vector range

	Monitor::init(filename);
	sys->register_monitor(this);
	src = NULL;
	nid = id;
	target_variable = state->data+nid;
	set_stop_time(10.0);
	ssize = sampling_interval/dt;
	outfile << setiosflags(ios::fixed) << setprecision(6);
}

void StateMonitor::init(SpikingGroup * source, NeuronID id, string statename, string filename, AurynTime stepsize)
{
	if ( !source->localrank(id) ) return; // do not register if neuron is not on the local rank


	Monitor::init(filename);
	sys->register_monitor(this);
	src = source;
	nid = src->global2rank(id);
	set_stop_time(10.0);

	if ( nid >= src->get_rank_size() ) {
		logger->msg("Error: StateMonitor trying to read from non-existing neuron.",ERROR);
		throw AurynStateVectorException();
	}

	ssize = stepsize;
	if ( ssize < 1 ) ssize = 1;

	if ( source->evolve_locally() ) {
		target_variable = src->get_state_vector(statename)->data+nid;
	} else {
		nid = src->get_rank_size() + 1;
	}
	outfile << setiosflags(ios::fixed) << setprecision(6);
}

StateMonitor::~StateMonitor()
{
}

void StateMonitor::propagate()
{
	if ( sys->get_clock() < t_stop && sys->get_clock()%ssize==0  ) {
		char buffer[255];
		int n = sprintf(buffer,"%f %f\n",sys->get_time(), *target_variable); 
		outfile.write(buffer,n); 
	}
}

void StateMonitor::set_stop_time(AurynDouble time)
{
	if (time < 0) {
		logger->msg("Warning: Negative stop times not supported -- ingoring.",WARNING);
	} 
	else t_stop = sys->get_clock() + time/dt;
}

void StateMonitor::record_for(AurynDouble time)
{
	set_stop_time(time);
}
