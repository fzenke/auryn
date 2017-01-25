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

#include "BinaryStateMonitor.h"

using namespace auryn;

const std::string BinaryStateMonitor::default_extension = "bst";

BinaryStateMonitor::BinaryStateMonitor(SpikingGroup * source, NeuronID id, std::string statename, std::string filename, AurynDouble sampling_interval)  : Monitor(filename, default_extension)
{

	if ( !source->localrank(id) ) return; // do not register if neuron is not on the local rank

	init(filename, sampling_interval);
	auryn::sys->register_device(this);
	src = source;
	nid = src->global2rank(id);

	if ( nid >= src->get_rank_size() ) {
		auryn::logger->msg("Error: BinaryStateMonitor trying to read from non-existing neuron.",ERROR);
		throw AurynStateVectorException();
	}

	// TODO test if state exists -- use find_state_vector instead .. 
	if ( source->evolve_locally() ) {
		target_variable = src->get_state_vector(statename)->data+nid;
	} else {
		nid = src->get_rank_size() + 1;
	}
}

BinaryStateMonitor::BinaryStateMonitor(auryn_vector_float * state, NeuronID id, std::string filename, AurynDouble sampling_interval): Monitor(filename, default_extension)
{
	if ( id >= state->size ) return; // do not register if neuron is out of vector range

	init(filename, sampling_interval);

	auryn::sys->register_device(this);
	src = NULL;
	nid = id;
	target_variable = state->data+nid;
	lastval = *target_variable;
}

BinaryStateMonitor::BinaryStateMonitor(Trace * trace, NeuronID id, std::string filename, AurynDouble sampling_interval): Monitor(filename, default_extension)
{
	if ( id >= trace->get_state_ptr()->size ) return; // do not register if neuron is out of vector range

	init(filename, sampling_interval);

	auryn::sys->register_device(this);
	src = NULL;
	nid = id;
	target_variable = trace->get_state_ptr()->data+nid;
	lastval = *target_variable;
}

void BinaryStateMonitor::open_output_file(std::string filename)
{
	if ( filename.empty() ) { // generate a default name
		filename = generate_filename();
	}

	outfile.open( filename.c_str(), std::ios::binary );
	if (!outfile) {
	  std::stringstream oss;
	  oss << "Can't open binary output file " << filename;
	  auryn::logger->msg(oss.str(),ERROR);
	  exit(1);
	}
}

void BinaryStateMonitor::init(std::string filename, AurynDouble sampling_interval)
{
	set_stop_time(10.0);
	ssize = sampling_interval/auryn_timestep;
	if ( ssize < 1 ) ssize = 1;

	enable_compression = true;
	lastval = 0.0;
	lastder = 0.0;


	// per convention the first entry contains
	// the number of timesteps per second
	// the neuronID field contains a tag 
	// encoding the version number
	StateValue_type headerFrame;
	headerFrame.time  = (AurynTime)(1.0/auryn_timestep);
	headerFrame.value = sys->build.tag_binary_state_monitor;
	outfile.write((char*)&headerFrame, sizeof(StateValue_type));
}



BinaryStateMonitor::~BinaryStateMonitor()
{
	AurynState value = *target_variable;
	AurynState deriv = value-lastval;
	if ( enable_compression && deriv==lastder ) { //terminate output with last value, but only if wasn't written already
		const AurynTime t = auryn::sys->get_clock()-ssize;
		write_frame(t, lastval);
	}

	outfile.close();
}


void BinaryStateMonitor::write_frame(const AurynTime time, const AurynState value)
{
	StateValue_type frame;
	frame.time  = time;
	frame.value = value;
	outfile.write((char*)&frame, sizeof(StateValue_type));
}

void BinaryStateMonitor::execute()
{
	if ( auryn::sys->get_clock() < t_stop && auryn::sys->get_clock()%ssize==0  ) {
		char buffer[255];
		if ( enable_compression && auryn::sys->get_clock()>0 ) {
			AurynState value = *target_variable;
			AurynState deriv = value-lastval;

			if ( deriv != lastder ) {
				const AurynTime t = auryn::sys->get_clock()-ssize;
				write_frame(t, lastval);
			}

			lastval = value;
			lastder = deriv;

		} else {
			const AurynTime t = auryn::sys->get_clock();
			write_frame(t, *target_variable);
		}
	}
}

void BinaryStateMonitor::set_stop_time(AurynDouble time)
{ 
	AurynDouble stoptime = std::min( time, std::numeric_limits<AurynTime>::max()*auryn_timestep );
	t_stop = stoptime/auryn_timestep;
}

void BinaryStateMonitor::record_for(AurynDouble time)
{
	if (time < 0) {
		auryn::logger->msg("Warning: Negative stop times not supported -- ingoring.",WARNING);
	} 
	else t_stop = auryn::sys->get_clock() + time/auryn_timestep;
	auryn::logger->debug("Set record for times for monitor.");
}
