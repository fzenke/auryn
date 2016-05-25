/* 
* Copyright 2014-2016 Friedemann Zenke
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

using namespace auryn;


StateMonitor::StateMonitor(SpikingGroup * source, NeuronID id, std::string statename, std::string filename, AurynDouble sampling_interval)  
{

	if ( !source->localrank(id) ) return; // do not register if neuron is not on the local rank

	init(filename, sampling_interval);
	auryn::sys->register_monitor(this);
	src = source;
	nid = src->global2rank(id);

	if ( nid >= src->get_rank_size() ) {
		auryn::logger->msg("Error: StateMonitor trying to read from non-existing neuron.",ERROR);
		throw AurynStateVectorException();
	}

	// TODO test if state exists -- use find_state_vector instead .. 
	if ( source->evolve_locally() ) {
		target_variable = src->get_state_vector(statename)->data+nid;
	} else {
		nid = src->get_rank_size() + 1;
	}
}

StateMonitor::StateMonitor(auryn_vector_float * state, NeuronID id, std::string filename, AurynDouble sampling_interval)
{
	if ( id >= state->size ) return; // do not register if neuron is out of vector range

	init(filename, sampling_interval);

	auryn::sys->register_monitor(this);
	src = NULL;
	nid = id;
	target_variable = state->data+nid;
}

StateMonitor::StateMonitor(EulerTrace * trace, NeuronID id, std::string filename, AurynDouble sampling_interval)
{
	if ( id >= trace->get_state_ptr()->size ) return; // do not register if neuron is out of vector range

	init(filename, sampling_interval);

	auryn::sys->register_monitor(this);
	src = NULL;
	nid = id;
	target_variable = trace->get_state_ptr()->data+nid;
}

void StateMonitor::init(std::string filename, AurynDouble sampling_interval)
{
	Monitor::init(filename);
	outfile << setiosflags(std::ios::fixed) << std::setprecision(6);

	set_stop_time(10.0);
	ssize = sampling_interval/dt;
	if ( ssize < 1 ) ssize = 1;

	enable_compression = true;
	lastval = 0.0;
	lastder = 0.0;
}

StateMonitor::~StateMonitor()
{
}

void StateMonitor::propagate()
{
	if ( auryn::sys->get_clock() < t_stop && auryn::sys->get_clock()%ssize==0  ) {
		char buffer[255];
		if ( enable_compression ) {
			AurynState value = *target_variable;
			AurynState deriv = value-lastval;

			if ( deriv != lastder ) {
				int n = sprintf(buffer,"%f %f\n",(auryn::sys->get_clock()-ssize)*dt, lastval); 
				outfile.write(buffer,n);
			}

			lastval = value;
			lastder = deriv;

		} else {
			int n = sprintf(buffer,"%f %f\n",auryn::sys->get_time(), *target_variable); 
			outfile.write(buffer,n); 
		}
	}
}

void StateMonitor::set_stop_time(AurynDouble time)
{ 
	AurynDouble stoptime = std::min( time, std::numeric_limits<AurynTime>::max()*dt );
	t_stop = stoptime/dt;
}

void StateMonitor::record_for(AurynDouble time)
{
	if (time < 0) {
		auryn::logger->msg("Warning: Negative stop times not supported -- ingoring.",WARNING);
	} 
	else t_stop = auryn::sys->get_clock() + time/dt;
	auryn::logger->debug("Set record for times for monitor.");
}
