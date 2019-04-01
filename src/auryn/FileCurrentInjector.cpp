/* 
* Copyright 2014-2019 Friedemann Zenke
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

#include "FileCurrentInjector.h"

using namespace auryn;

FileCurrentInjector::FileCurrentInjector(NeuronGroup * target, std::string time_series_file, std::string neuron_state_name ) : CurrentInjector(target, neuron_state_name, 0.0 )
{
	target_neuron_ids = new std::vector<NeuronID>;
	current_time_series = new std::vector<AurynState>;

	mode = ALL;
	loop = false;
	set_loop_grid(1.0);
	load_time_series_file(time_series_file);
}



void FileCurrentInjector::free( ) 
{
	delete target_neuron_ids;
	delete current_time_series;
}


FileCurrentInjector::~FileCurrentInjector()
{
	free();
}


AurynState FileCurrentInjector::get_current_current_value()
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

void FileCurrentInjector::execute()
{
	if ( dst->evolve_locally() ) {
		AurynState cur = get_current_current_value();
		switch ( mode ) {
			case LIST:
				currents->set_all(0.0);
				for ( int i = 0 ; i < target_neuron_ids->size() ; ++i ) {
					currents->set(target_neuron_ids->at(i),cur);
				}
				break;
			case ALL:
			default:
				currents->set_all(cur);
		}
		CurrentInjector::execute();
	}
}

void FileCurrentInjector::set_loop_grid(double grid)
{
	if (grid<=0.0) {
		logger->error("FileCurrentInjector:: Cannot set non-positive loop grid. Loop grid unchanged!");
	} else {
		loop_grid = grid/auryn_timestep;
	}
}

void FileCurrentInjector::load_time_series_file(std::string filename)
{
	std::ifstream inputfile;
	inputfile.open(filename.c_str(),std::ifstream::in);
	if (!inputfile) {
	  std::cerr << "Can't open input file " << filename << std::endl;
	  throw AurynOpenFileException();
	}

	current_time_series->clear();

	// read the time series file and interpolate missing time points linearly
	AurynTime curtime = 0;
	AurynTime ltime = 0;
	AurynTime ntime = 0;
	double lc = 0.0;
	double nc = 0.0;
	char buffer[255];
	while ( inputfile.getline(buffer, 256) ) {
		std::stringstream line ( buffer );

		// store last data point
		ltime = ntime; 
		lc = nc;

		// read new data point
		double t;
		line >> t;
		ntime = (AurynTime) (t/auryn_timestep+0.5);
		line >> nc;

		while ( curtime < ntime || inputfile.eof() ) { 
			double inter_current = lc + 1.0*(curtime-ltime)/(ntime-ltime)*(nc-lc);
			// std::cout << curtime << " " << inter_current << std::endl;
			current_time_series->push_back(inter_current);
			curtime++;
		} 
	}
	current_time_series->push_back(lc); // terminates with last value

	inputfile.close();
}
