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

#include "auryn.h"
#include <iostream>
#include <fstream>

using namespace std;

namespace po = boost::program_options;


/*! Perform binary search on ifstream to extract frame number
 * from a target time reference that should be given in discrete time. */
AurynLong FindFrame( ifstream * file, AurynTime target )
{
	// get number of elements
	file->seekg (0, file->end);
	AurynLong num_of_frames = file->tellg()/sizeof(SpikeEvent_type);

	AurynLong lo = 1; // first frame is used for header
	AurynLong hi = num_of_frames;

	while ( lo+1 < hi ) {
		AurynLong pivot = lo + (hi-lo)/2;
		file->seekg (pivot*sizeof(SpikeEvent_type), file->beg);

		SpikeEvent_type spike_data;
		file->read((char*)&spike_data, sizeof(SpikeEvent_type));

		if ( spike_data.time < target ) lo = pivot;
		else hi = pivot;
	}

	//cout << lo << " " << hi << endl;

	return lo;
}

int main(int ac, char* av[]) 
{
	string input_filename = "";
	string output_file_name = "";
	double from_time = 0.0;
	double to_time   = 100.0;
	double seconds_to_extract_from_end = -1.0; // negative means disabled
	NeuronID maxid = std::numeric_limits<NeuronID>::max();

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			("version", "show version information")
			("file", po::value<string>(), "input file")
			("output", po::value<string>(), "output file (output to stout if not given)")
			("start", po::value<double>(), "start time in seconds")
			("to", po::value<double>(), "to time in seconds")
			("last", po::value<double>(), "last x seconds (overrides start/end)")
			("maxid", po::value<NeuronID>(), "maximum neuron id to extract")
			;

		po::variables_map vm;        
		po::store(po::parse_command_line(ac, av, desc), vm);
		po::notify(vm);    

		if (vm.count("help")) {
			cout << desc << "\n";
			return 1;
		}

		if (vm.count("version")) {
			cout << "Auryn Binary Extract version " 
				 << AURYNVERSION << "." 
				 << AURYNSUBVERSION << "."
				 << AURYNREVISION << "\n";
			return EXIT_SUCCESS;
		}

		if (vm.count("file")) {
			input_filename = vm["file"].as<string>();
		} 

		if (vm.count("output")) {
			output_file_name = vm["output"].as<string>();
		} 

		if (vm.count("start")) {
			from_time = vm["start"].as<double>();
        } 

        if (vm.count("to")) {
			to_time = vm["to"].as<double>();
        } 

        if (vm.count("last")) {
			seconds_to_extract_from_end = vm["last"].as<double>();
        } 

        if (vm.count("maxid")) {
			maxid = vm["maxid"].as<NeuronID>();
        } 
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }


	ifstream * input;

	input = new ifstream( input_filename.c_str(), ios::binary );
	if (!(*input)) {
		std::cerr << "Unable to open input file" << endl;
		exit(EXIT_FAILURE);
	}

	// get length of the file
	SpikeEvent_type spike_data;
	input->seekg (0, input->end);
	AurynLong num_events = input->tellg()/sizeof(SpikeEvent_type)-1;

	// read first entry to infer dt 
	input->seekg (0, input->beg);
	input->read((char*)&spike_data, sizeof(SpikeEvent_type));
	double dt = 1.0/spike_data.time;

	// do some version checking
	NeuronID tag = spike_data.neuronID;
	if ( tag/1000 != tag_binary_spike_monitor/1000 ) {
		cerr << "Header not recognized. " 
			"Not a binary Auryn monitor file?" 
			 << endl;
		exit(EXIT_FAILURE);
	}

	// read out last time
	input->seekg (num_events*sizeof(SpikeEvent_type), input->beg);
	input->read((char*)&spike_data, sizeof(SpikeEvent_type));
	double last_time = spike_data.time*dt;


	if ( tag != tag_binary_spike_monitor ) {
		cerr << "# Either the Auryn version does not match "
			"the version of this tool or this is not a spike "
			"raster file." << endl; 
		// TODO tell user if it is a state file
	}

	if ( seconds_to_extract_from_end > 0 ) {
		to_time = last_time;
		from_time = to_time-seconds_to_extract_from_end;
	}

	if ( from_time < 0 ) from_time = 0.0 ;

	if ( from_time > to_time || from_time < 0 ) {
		cerr << "Times must be positive and start "
			"time needs to be < to time." << endl;
		exit(EXIT_FAILURE);
	}

	// compute start and end frames
	AurynLong start_frame = FindFrame(input, from_time/dt);
	AurynTime to_auryn_time = to_time/dt;


#ifdef DEBUG
	cerr << "# Timestep: " << dt << endl;
	cerr << "# Maxid: " << maxid << endl;
	cerr << "# Sizeof SpikeEvent struct: " << sizeof(SpikeEvent_type) << endl;
	cerr << "# Time of last event in file: " << last_time << endl;
	cerr << "# From time: " << from_time << endl;
	cerr << "# To time: " << to_time << endl;
	cerr << "# Start frame: " << start_frame << endl;
#endif // DEBUG


	// prepare input stream
	input->seekg (start_frame*sizeof(SpikeEvent_type), input->beg);
	input->clear();
	if(!output_file_name.empty()) {
		std::ofstream of;
		of.open( output_file_name.c_str(), std::ofstream::out );
		while ( true ) {
			input->read((char*)&spike_data, sizeof(SpikeEvent_type));
			if ( spike_data.time >= to_auryn_time || input->eof() ) break;
			if ( spike_data.neuronID > maxid) continue;
			of << spike_data.time*dt << " " << spike_data.neuronID << "\n";
		}
		of.close();
	} else {
		while ( true ) {
			input->read((char*)&spike_data, sizeof(SpikeEvent_type));
			if ( spike_data.time >= to_auryn_time || input->eof() ) break;
			if ( spike_data.neuronID > maxid) continue;
			cout << spike_data.time*dt << " " << spike_data.neuronID << "\n";
		}
	}
	
	input->close();
	return EXIT_SUCCESS;
}
