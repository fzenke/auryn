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

#include "FileInputGroup.h"

using namespace auryn;


void FileInputGroup::init()
{
	auryn::sys->register_spiking_group(this);
	active = true;
	loop_grid_size = 1;
	reset_time = 0;
}

FileInputGroup::FileInputGroup( NeuronID n ) : SpikingGroup(n, RANKLOCK ) 
{
	playinloop = false;
	time_delay = 0;
	time_offset = 0;
	init();
}

FileInputGroup::FileInputGroup(NeuronID n, std::string filename, 
		bool loop, AurynFloat delay) 
: SpikingGroup( n , RANKLOCK )
{
	playinloop = loop;
	time_delay = (AurynTime) (delay/auryn_timestep);
	time_offset = 0;
	init();
	load_spikes(filename);
}

FileInputGroup::~FileInputGroup()
{
}

bool time_compare (SpikeEvent_type a,SpikeEvent_type b) { return (a.time<b.time); }

void FileInputGroup::load_spikes(std::string filename)
{
	std::ifstream spkfile;
	input_spikes.clear();

	if ( evolve_locally() ) {
		spkfile.open(filename.c_str(),std::ifstream::in);
		if (!spkfile) {
			std::cerr << "Can't open input file " << filename << std::endl;
			std::exit(1);
		}
	}

	char buffer[255];
	while ( spkfile.getline(buffer, 256) ) {
		SpikeEvent_type event;
		std::stringstream line ( buffer ) ;
		double t_tmp;
		line >> t_tmp;
		event.time = t_tmp/auryn_timestep;
		line >> event.neuronID;
		if ( localrank(event.neuronID) ) {
			input_spikes.push_back(event);
			// std::cout << event.time << std::endl;
		}
	}
	spkfile.close();

	sort_spikes();

	std::stringstream oss;
	oss << get_log_name() << ":: Finished loading " << input_spikes.size() 
		<< " spike events";
	logger->info(oss.str());

	spike_iter = input_spikes.begin();
}

void FileInputGroup::sort_spikes()
{
	std::sort (input_spikes.begin(), input_spikes.end(), time_compare);
}


AurynTime FileInputGroup::get_offset_clock() 
{
	return sys->get_clock() - time_offset;
}

AurynTime FileInputGroup::get_next_grid_point( AurynTime time ) 
{
	AurynTime result = time+time_delay;
	if ( result%loop_grid_size ) { // align to temporal grid
		result = (result/loop_grid_size+1)*loop_grid_size;
	}
	return result;
}

void FileInputGroup::evolve()
{
	if (active && input_spikes.size()) {
		// when reset_time is reached reset the spike_iterator to The beginning and update time offset
		if ( sys->get_clock() == reset_time ) {
			spike_iter = input_spikes.begin(); 
			time_offset = sys->get_clock();
			// std::cout << "set to" << reset_time*auryn_timestep << " " << time_offset << std::endl;
		}

		while ( spike_iter != input_spikes.end() && (*spike_iter).time <= get_offset_clock() ) {
			spikes->push_back((*spike_iter).neuronID);
			++spike_iter;
			// std::cout << "spike " << sys->get_time() << std::endl;
		}

		// TODO Fix the bug which eats the first spike
		if ( spike_iter==input_spikes.end() && reset_time < sys->get_clock() && playinloop ) { // at last spike on file set new reset time
			// schedule reset for next grid point after delay
			reset_time = get_next_grid_point(sys->get_clock());
			// std::cout << "set rt" << reset_time << std::endl;
		}
	}
}


void FileInputGroup::set_loop_grid(AurynDouble grid_size)
{
	playinloop = true;
	if ( grid_size > 0.0 ) {
		loop_grid_size = 1.0/auryn_timestep*grid_size;
		if ( loop_grid_size == 0 ) loop_grid_size = 1;
	}
}

void FileInputGroup::add_spike(double spiketime, NeuronID neuron_id)
{
		SpikeEvent_type event;
		event.time = spiketime/auryn_timestep;
		event.neuronID = neuron_id;
		input_spikes.push_back(event);
		sort_spikes();
}

