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

#define DEBUG

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

	AurynLong lo = 0;
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
	double start_time = 0.0;
	double end_time   = 100.0;
	double seconds_to_extract_from_end = -1.0; // negative means disabled

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("file", po::value<string>(), "input file")
            ("output", po::value<string>(), "output file (output to stout if not given)")
            ("start", po::value<double>(), "start time in seconds")
            ("end", po::value<double>(), "end time in seconds")
            ("last", po::value<double>(), "last x seconds (overrides start/end)")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        if (vm.count("file")) {
			input_filename = vm["file"].as<string>();
        } 

        if (vm.count("output")) {
			output_file_name = vm["output"].as<string>();
        } 

        if (vm.count("start")) {
			start_time = vm["start"].as<double>();
        } 

        if (vm.count("end")) {
			end_time = vm["end"].as<double>();
        } 

        if (vm.count("last")) {
			seconds_to_extract_from_end = vm["last"].as<double>();
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
	AurynLong num_events = input->tellg()/sizeof(SpikeEvent_type);

	// read first entry to infer dt 
	input->seekg (0, input->beg);
	input->read((char*)&spike_data, sizeof(SpikeEvent_type));
	double dt = 1.0/spike_data.time;
	NeuronID group_size = spike_data.neuronID;

	// TODO do some version checking

	// read out last time
	input->seekg (num_events*sizeof(SpikeEvent_type), input->beg);
	input->read((char*)&spike_data, sizeof(SpikeEvent_type));
	double last_time = spike_data.time*dt;

	if ( seconds_to_extract_from_end > 0 ) {
		end_time = last_time;
		start_time = end_time-seconds_to_extract_from_end;
	}

	if ( start_time < 0 ) start_time = 0.0 ;

	if ( start_time > end_time || start_time < 0 ) {
		cerr << "Times must be positive and start "
			"time needs to be < end time." << endl;
		exit(EXIT_FAILURE);
	}

	// compute start and end frames
	AurynLong start_frame = FindFrame(input, start_time/dt);
	AurynTime end_auryn_time = end_time/dt;


#ifdef DEBUG
	cerr << "# Timestep: " << dt << endl;
	cerr << "# Sizeof SpikeEvent struct: " << sizeof(SpikeEvent_type) << endl;
	cerr << "# Time of last event in file: " << last_time << endl;
	cerr << "# Start time: " << start_time << endl;
	cerr << "# End time: " << end_time << endl;
	cerr << "# Start frame: " << start_frame << endl;
#endif // DEBUG


	// prepare input stream
	input->seekg (start_frame*sizeof(SpikeEvent_type)+1, input->beg);
	if(!output_file_name.empty()) {
		std::ofstream of;
		of.open( output_file_name.c_str(), std::ofstream::out );
		while (!input->eof()) {

			input->read((char*)&spike_data, sizeof(SpikeEvent_type));
			if ( spike_data.time > end_auryn_time ) break;
			of << spike_data.time*dt << " " << spike_data.neuronID << "\n";
		}
		of.close();
	} else {
		while (!input->eof()) {
			input->read((char*)&spike_data, sizeof(SpikeEvent_type));
			if ( spike_data.time > end_auryn_time ) break;
			cout << spike_data.time*dt << " " << spike_data.neuronID << "\n";
		}
	}
	
	input->close();
	return EXIT_SUCCESS;
}
