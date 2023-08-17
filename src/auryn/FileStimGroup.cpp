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

#include "FileStimGroup.h"

using namespace auryn;

boost::mt19937 FileStimGroup::order_gen = boost::mt19937(); 
boost::uniform_01<boost::mt19937> FileStimGroup::order_die = boost::uniform_01<boost::mt19937> (order_gen);

void FileStimGroup::init()
{
	auryn::sys->register_spiking_group(this);
	active = true;
	loop_grid_size = 1;
	reset_time = 0;

	seed(42);


	set_stimulation_mode(RANDOM); // TODO make adjustable
}

void FileStimGroup::set_stimulation_mode( StimulusGroupModeType mode ) {
	stimulus_order = mode ;
}

FileStimGroup::FileStimGroup( NeuronID n ) : SpikingGroup(n, RANKLOCK ) 
{
	playinloop = false;
	time_delay = 0;
	time_offset = 0;
	current_pattern_index = 0;
	init();
}

FileStimGroup::FileStimGroup(NeuronID n, std::string filename, 
		bool loop, AurynFloat delay) 
: SpikingGroup( n , RANKLOCK )
{
	init();
	playinloop = loop;
	if ( playinloop ) set_loop_grid(auryn_timestep);
	time_delay = (AurynTime) (delay/auryn_timestep);
	time_offset = 0;
	load_spikes(filename);
}

FileStimGroup::~FileStimGroup()
{
	clear_input_patterns();
}

bool time_compare (SpikeEvent_type a,SpikeEvent_type b) { return (a.time<b.time); }


void FileStimGroup::clear_input_patterns() 
{
	for ( unsigned int i = 0 ; i<input_patterns.size() ; ++i ) {
		delete input_patterns.at(i);
	}
	input_patterns.clear();
	current_pattern_index = 0;
}

void FileStimGroup::load_spikes(std::string filename)
{
	clear_input_patterns();
	add_spikes(filename);
}

void FileStimGroup::add_spikes(std::string filename)
{
	std::ifstream spkfile;

	if ( evolve_locally() ) {
		spkfile.open(filename.c_str(),std::ifstream::in);
		if (!spkfile) {
			std::cerr << "Can't open input file " << filename << std::endl;
			std::exit(1);
		}

		current_pattern = new std::vector< SpikeEvent_type >;
		char buffer[255];
		std::string line_;
		while ( spkfile.getline(buffer, 256) ) {
			line_.clear(); // TODO make this less hacky
			line_ = buffer;
			if ( line_[0] == '#' ) continue; // skip comments
			if ( line_ == "" and current_pattern->size()>0 ) { // empty line triggers new pattern
				sort_spikes(current_pattern);
				input_patterns.push_back(current_pattern);
				current_pattern = new std::vector< SpikeEvent_type >;
				continue;
			}

			// assume we have a line with an event
			SpikeEvent_type event;
			std::stringstream line ( buffer ) ;
			double t_tmp;
			line >> t_tmp;
			event.time = round(t_tmp/auryn_timestep);
			line >> event.neuronID;
			if ( localrank(event.neuronID) ) current_pattern->push_back(event);
		}
		spkfile.close();

		// store last pattern if any
		if ( current_pattern->size()>0 ) { 
			sort_spikes(current_pattern);
			input_patterns.push_back(current_pattern);
		}


		std::stringstream oss;
		oss << get_log_name() << ":: Finished loading " << input_patterns.size() 
			<< " input patterns";
		logger->info(oss.str());

		current_pattern = input_patterns.at(0);
		spike_iter = current_pattern->end(); 
	}
}

void FileStimGroup::sort_spikes( std::vector< SpikeEvent_type > * pattern )
{
	std::sort (pattern->begin(), pattern->end(), time_compare);
}


AurynTime FileStimGroup::get_offset_clock() 
{
	return sys->get_clock() - time_offset;
}

AurynTime FileStimGroup::get_next_grid_point( AurynTime time ) 
{
	AurynTime result = time+time_delay;
	if ( result%loop_grid_size ) { // align to temporal grid
		result = (result/loop_grid_size+1)*loop_grid_size;
	}
	return result+1;
}

void FileStimGroup::evolve()
{
	if ( active && input_patterns.size() ) {
		// when reset_time is reached reset the spike_iterator to The beginning and update time offset
		if ( sys->get_clock() == reset_time ) {
			spike_iter = current_pattern->begin(); 
			time_offset = sys->get_clock();
			// std::cout << "set to" << reset_time*auryn_timestep << " " << time_offset << std::endl;
		}

		while ( spike_iter != current_pattern->end() && (*spike_iter).time <= get_offset_clock() ) {
			spikes->push_back((*spike_iter).neuronID);
			++spike_iter;
			// std::cout << "spike " << sys->get_time() << std::endl;
		}

		if ( spike_iter==current_pattern->end() && reset_time < sys->get_clock() && playinloop ) { // at last spike on file set new reset time

			// chooses stimulus according to schema specified in stimulusmode
			switch ( stimulus_order ) {
				case RANDOM:
					double draw;
					draw = order_die();
					current_pattern_index = draw*input_patterns.size();
				break;
				case SEQUENTIAL:
					current_pattern_index = (current_pattern_index+1)%input_patterns.size();
				break;
				case SEQUENTIAL_REV:
					if ( current_pattern_index == 0 ) 
						current_pattern_index = input_patterns.size() - 1 ;
					else
						--current_pattern_index;
				break;
				case MANUAL:
				default:
				break;
			}


			// std::cout << "new pattern " << current_pattern_index << std::endl;
			current_pattern = input_patterns.at(current_pattern_index);
			reset_time = get_next_grid_point(sys->get_clock());

			std::stringstream oss;
			oss << get_log_name() << ":: Selected next pattern " << current_pattern_index << " at " << sys->get_time() 
				<< "s. Current grid point " << sys->get_clock()
				<< ", next grid point " << reset_time;
			logger->debug(oss.str());
		}
	}
}

void FileStimGroup::seed(int rndseed)
{
	unsigned int rnd = rndseed + sys->get_synced_seed(); // most be same on all ranks
	order_gen.seed(rnd); 

	// this is here because the seeding above alone does not seem to do anything
	// also need to overwrite the dist operator because it makes of copy of the
	// generator
	// see http://www.bnikolic.co.uk/blog/cpp-boost-uniform01.html
	order_die = boost::uniform_01<boost::mt19937> (order_gen);
}

void FileStimGroup::set_loop_grid(AurynDouble grid_size)
{
	playinloop = true;
	if ( grid_size > 0.0 ) {
		loop_grid_size = 1.0/auryn_timestep*grid_size;
		if ( loop_grid_size == 0 ) loop_grid_size = 1;
	}
}

void FileStimGroup::add_spike(double spiketime, NeuronID neuron_id)
{
		SpikeEvent_type event;
		event.time = spiketime/auryn_timestep;
		event.neuronID = neuron_id;
		current_pattern->push_back(event);
		sort_spikes(); 
}

