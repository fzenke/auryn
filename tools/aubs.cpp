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
#include <vector>
#include "auryn/AurynVersion.h"

using namespace auryn;

namespace po = boost::program_options;

/*! \file 
 * Implements helper program to decode files written with BinaryStateMonitor
 */


/*! Perform binary search on ifstream to extract frame number
 * from a target time reference that should be given in discrete time. */
AurynLong find_frame( std::ifstream * file, AurynTime target )
{
	// get number of elements
	file->seekg (0, file->end);
	AurynLong num_of_frames = file->tellg()/sizeof(StateValue_type);

	AurynLong lo = 1; // first frame is used for header
	AurynLong hi = num_of_frames;

	while ( lo+1 < hi ) {
		AurynLong pivot = lo + (hi-lo)/2;
		file->seekg (pivot*sizeof(StateValue_type), file->beg);

		StateValue_type state_data;
		file->read((char*)&state_data, sizeof(StateValue_type));

		if ( state_data.time < target ) lo = pivot;
		else hi = pivot;
	}

	return hi;
}


void read_header( std::ifstream * input, double& dt, AurynLong num_events, double& last_time, std::string filename )
{
	// get length of the file
	StateValue_type state_data;
	input->seekg (0, input->end);
	num_events = input->tellg()/sizeof(StateValue_type)-1;

	// read first entry to infer dt 
	input->seekg (0, input->beg);
	input->read((char*)&state_data, sizeof(StateValue_type));
	dt = 1.0/state_data.time;

	// do some version checking
	AurynVersion build;
	AurynState tag = state_data.value;
	if ( (int)tag/1000 != (int)(build.tag_binary_state_monitor)/1000 ) {
		std::cerr << "Header not recognized. " 
			"Not a BinaryStateMonitor file?" 
			 << std::endl;
		exit(EXIT_FAILURE);
	}

	if ( tag != build.tag_binary_state_monitor ) {
		std::cerr << "# Warning: Either the Auryn version does not match "
			"the version of this tool or this is not a BinaryStateMonitor "
			"file." << std::endl; 
		// TODO tell user if it is a state file
	}

	// read out last time
	input->seekg (num_events*sizeof(StateValue_type), input->beg);
	input->read((char*)&state_data, sizeof(StateValue_type));
	last_time = (state_data.time+1)*dt; 
	// places last_time _behind_ the last time because we are the 
	// exclusive interval end
}

int main(int ac, char* av[]) 
{
	std::string input_filename;
	std::ifstream * input;

	std::string output_file_name = "";
	double from_time = 0.0;
	double to_time   = -1.0;
	double seconds_to_extract_from_end = -1.0; // negative means disabled
	// one more decimal than neede to show values are not rounded
	bool debug_output = false;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("version,v", "show version information")
			("debug,d", "show verbose debug output")
			("input,i", po::value<std::string>(), "input file")
			("output,o", po::value<std::string>(), "output file (output to stout if not given)")
			("from,f", po::value<double>(), "'from time' in seconds")
			("to,t", po::value<double>(), "'to time' in seconds")
			("last,l", po::value<double>(), "last x seconds (overrides from/to)")
			;

		po::variables_map vm;        
		po::store(po::parse_command_line(ac, av, desc), vm);
		po::notify(vm);    

		if (vm.count("help")) {
			std::cout << desc << "\n";
			return 1;
		}

		if (vm.count("version")) {
			AurynVersion build;
			std::cout << "Auryn Binary Extract version " 
				 << build.version << "." 
				 << build.subversion << "."
				 << build.revision_number << "\n";
			return EXIT_SUCCESS;
		}

		if (vm.count("debug")) {
			debug_output = true;
		}

		if (vm.count("input")) {
			input_filename = vm["input"].as<std::string>();
		} 

		if (vm.count("output")) {
			output_file_name = vm["output"].as<std::string>();
		} 

		if (vm.count("from")) {
			from_time = vm["from"].as<double>();
        } 

        if (vm.count("to")) {
			to_time = vm["to"].as<double>();
        } 

        if (vm.count("last")) {
			seconds_to_extract_from_end = vm["last"].as<double>();
        } 

    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }

	double last_time = 0.0;
	double dt = 0.0;

	if ( input_filename.empty() ) {
		std::cerr << "Missing input file." << std::endl;
		exit(EXIT_FAILURE);
	}

	input = new std::ifstream( input_filename.c_str(), std::ios::binary );
	if (!(*input)) {
		std::cerr << "Unable to open input file " 
			<< input_filename
			<< std::endl;
		exit(EXIT_FAILURE);
	}

	double tmp_last_time = 0;
	double tmp_dt = 0;
	AurynLong tmp_num_events = 0;
	read_header( input, tmp_dt, tmp_num_events, tmp_last_time, input_filename );

	if ( debug_output ) {
		std::cerr << "# Last frame in file:" << tmp_num_events << std::endl;
	}

	if ( dt == 0 ) {
		dt = tmp_dt;
	} else {
		if ( dt != tmp_dt ) { // should not happen
			std::cerr << "Not all input file headers match." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	if ( tmp_last_time > last_time ) {
		last_time = tmp_last_time;
	}


	if ( to_time < 0 ) {
		to_time = last_time;
	}

	if ( seconds_to_extract_from_end > 0 ) {
		from_time = to_time-seconds_to_extract_from_end;
	}

	if ( from_time < 0 ) from_time = 0.0 ;

	if ( from_time > to_time || from_time < 0 ) {
		std::cerr << "Times must be positive and start "
			"time needs to be < to time." << std::endl;
		exit(EXIT_FAILURE);
	}

	// translate second times into auryn time
	AurynTime to_auryn_time = to_time/dt;


	if ( debug_output ) {
		std::cerr << "# Timestep: " << dt << std::endl;
		std::cerr << "# Sizeof SpikeEvent struct: " << sizeof(StateValue_type) << std::endl;
		std::cerr << "# Time of last event in files: " << last_time << std::endl;
		std::cerr << "# From time: " << from_time << std::endl;
		std::cerr << "# To time: " << to_time << std::endl;
	}


	// set stream to respetive start frame
	
	// compute start and end frames
	AurynLong start_frame = find_frame(input, from_time/dt);

	// prepare input stream
	input->seekg (start_frame*sizeof(StateValue_type), input->beg);
	input->clear();
	if ( debug_output ) {
		std::cerr << "# Start frame stream: " 
			<< start_frame << std::endl;
	}


	StateValue_type frame;
	input->read((char*)&frame, sizeof(StateValue_type));

	AurynTime time_reference = from_time/dt;
	int decimal_places = -std::log(dt)/std::log(10)+2; 

	// open output filestream if needed
	std::ofstream of;
	bool write_to_stdout = true;
 	if( !output_file_name.empty() ) {
		write_to_stdout = false;
 		of.open( output_file_name.c_str(), std::ofstream::out );
		of << std::fixed << std::setprecision(decimal_places);
	}
	// sets output format to right number of decimal places
	std::cout << std::fixed << std::setprecision(decimal_places);

	while ( true ) {
		time_reference = frame.time;
		if ( time_reference >= to_auryn_time || input->eof() ) break;

		if ( debug_output && false ) {
			std::cout << "# time_reference " << time_reference << std::endl;
		}

		// output from next_stream
		while ( frame.time <= time_reference && !input->eof() ) {
			if ( write_to_stdout ) 
				std::cout << std::fixed << std::setprecision(decimal_places) 
					<< frame.time*dt << " " 
					<< std::scientific // << std::setprecision(8)
					<< frame.value << "\n";
			else 
				of << std::fixed << std::setprecision(decimal_places) 
					<< frame.time*dt << " " 
					<< std::scientific // << std::setprecision(8) 
					<< frame.value << "\n";
			input->read((char*)&frame, sizeof(StateValue_type));
		}
	}

	if ( !write_to_stdout ) 
 		of.close();

	// close input stream
	input->close();

	return EXIT_SUCCESS;
}
