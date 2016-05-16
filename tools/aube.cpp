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

using namespace auryn;

namespace po = boost::program_options;


/*! Perform binary search on ifstream to extract frame number
 * from a target time reference that should be given in discrete time. */
AurynLong find_frame( std::ifstream * file, AurynTime target )
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

	return hi;
}


void read_header( std::ifstream * input, double& dt, AurynLong num_events, double& last_time, std::string filename )
{
	// get length of the file
	SpikeEvent_type spike_data;
	input->seekg (0, input->end);
	num_events = input->tellg()/sizeof(SpikeEvent_type)-1;

	// read first entry to infer dt 
	input->seekg (0, input->beg);
	input->read((char*)&spike_data, sizeof(SpikeEvent_type));
	dt = 1.0/spike_data.time;

	// do some version checking
	NeuronID tag = spike_data.neuronID;
	if ( tag/1000 != tag_binary_spike_monitor/1000 ) {
		std::cerr << "Header not recognized. " 
			"Not a binary Auryn monitor file?" 
			 << std::endl;
		exit(EXIT_FAILURE);
	}

	if ( tag != tag_binary_spike_monitor ) {
		std::cerr << "# Warning: Either the Auryn version does not match "
			"the version of this tool or this is not a spike "
			"raster file." << std::endl; 
		// TODO tell user if it is a state file
	}

	// read out last time
	input->seekg (num_events*sizeof(SpikeEvent_type), input->beg);
	input->read((char*)&spike_data, sizeof(SpikeEvent_type));
	last_time = (spike_data.time+1)*dt; 
	// places last_time _behind_ the last time because we are the 
	// exclusive interval end
}

int main(int ac, char* av[]) 
{
	std::vector<std::string> input_filenames;
	std::vector<std::ifstream*> inputs;

	std::string output_file_name = "";
	double from_time = 0.0;
	double to_time   = -1.0;
	double seconds_to_extract_from_end = -1.0; // negative means disabled
	NeuronID maxid = std::numeric_limits<NeuronID>::max();
	// one more decimal than neede to show values are not rounded
	int decimal_places = -std::log(dt)/std::log(10)+2; 
	bool debug_output = false;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("version,v", "show version information")
			("debug,d", "show verbose debug output")
			("inputs,i", po::value< std::vector<std::string> >()->multitoken(), "input files")
			("output,o", po::value<std::string>(), "output file (output to stout if not given)")
			("from,f", po::value<double>(), "'from time' in seconds")
			("to,t", po::value<double>(), "'to time' in seconds")
			("last,l", po::value<double>(), "last x seconds (overrides from/to)")
			("maxid,m", po::value<NeuronID>(), "maximum neuron id to extract")
			;

		po::variables_map vm;        
		po::store(po::parse_command_line(ac, av, desc), vm);
		po::notify(vm);    

		if (vm.count("help")) {
			std::cout << desc << "\n";
			return 1;
		}

		if (vm.count("version")) {
			std::cout << "Auryn Binary Extract version " 
				 << AURYNVERSION << "." 
				 << AURYNSUBVERSION << "."
				 << AURYNREVISION << "\n";
			return EXIT_SUCCESS;
		}

		if (vm.count("debug")) {
			debug_output = true;
		}

		if (vm.count("inputs")) {
			 input_filenames = vm["inputs"].as< std::vector<std::string> >();
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

        if (vm.count("maxid")) {
			maxid = vm["maxid"].as<NeuronID>();
        } 
    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }

	if ( debug_output ) 
		std::cout << "# Number of input files " << input_filenames.size() << std::endl;

	double last_time = 0.0;
	double dt = 0.0;

	if ( input_filenames.size() == 0 ) {
		std::cerr << "Missing input file." << std::endl;
		exit(EXIT_FAILURE);
	}

	for ( int i = 0 ; i < input_filenames.size() ; ++i ) {
		std::ifstream * tmp = new std::ifstream( input_filenames[i].c_str(), std::ios::binary );
		inputs.push_back(tmp);
		if (!(*tmp)) {
			std::cerr << "Unable to open input file " 
				<< input_filenames[i]
				<< std::endl;
			exit(EXIT_FAILURE);
		}

		double tmp_last_time = 0;
		double tmp_dt = 0;
		AurynLong tmp_num_events = 0;
		read_header( tmp, tmp_dt, tmp_num_events, tmp_last_time, input_filenames[i] );

		if ( debug_output ) {
			std::cerr << "# Last frame in file " << i << ": " << tmp_num_events << std::endl;
		}

		if ( dt == 0 ) {
			dt = tmp_dt;
		} else {
			if ( dt != tmp_dt ) { // should not happen
				std::cerr << "Not all input file headers match." << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		if ( tmp_last_time > last_time )
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
		std::cerr << "# Maxid: " << maxid << std::endl;
		std::cerr << "# Sizeof SpikeEvent struct: " << sizeof(SpikeEvent_type) << std::endl;
		std::cerr << "# Time of last event in files: " << last_time << std::endl;
		std::cerr << "# From time: " << from_time << std::endl;
		std::cerr << "# To time: " << to_time << std::endl;
	}


	// set all streams to respetive start frame
	for ( int i = 0 ; i < inputs.size() ; ++i ) {
		// compute start and end frames
		AurynLong start_frame = find_frame(inputs[i], from_time/dt);

		// prepare input stream
		inputs[i]->seekg (start_frame*sizeof(SpikeEvent_type), inputs[i]->beg);
		inputs[i]->clear();
		if ( debug_output ) {
			std::cerr << "# Start frame stream " 
				<< i << ": " 
				<< start_frame << std::endl;
		}
	}


	// read first frames from all files
	std::vector<SpikeEvent_type> frames(inputs.size());
	for ( int i = 0 ; i < frames.size() ; ++i ) {
		inputs[i]->read((char*)&frames[i], sizeof(SpikeEvent_type));
	}

	AurynTime time_reference = from_time/dt;

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
		// find smallest time reference
		int current_stream = 0;
		AurynLong mintime = std::numeric_limits<AurynLong>::max();
		bool eofs = true;
		for ( int i = 0 ; i < frames.size() ; ++i ) {
			eofs = eofs && inputs[i]->eof();
			if ( inputs[i]->eof() ) continue;
			if ( frames[i].time < mintime ) { 
				mintime = frames[i].time;
				current_stream = i;
			}
		}

		time_reference = mintime;
		if ( time_reference >= to_auryn_time || eofs ) break;

		if ( debug_output && false ) {
			std::cout << "# current_stream " << current_stream << std::endl;
			std::cout << "# time_reference " << time_reference << std::endl;
		}

		// output from next_stream
		while ( frames[current_stream].time <= time_reference && !inputs[current_stream]->eof() ) {
			if ( frames[current_stream].neuronID < maxid) {
				if ( write_to_stdout ) 
					std::cout << frames[current_stream].time*dt << " " << frames[current_stream].neuronID << "\n";
				else 
					of << frames[current_stream].time*dt << " " << frames[current_stream].neuronID << "\n";

			}
			inputs[current_stream]->read((char*)&frames[current_stream], sizeof(SpikeEvent_type));
		}
	}

	if ( !write_to_stdout ) 
 		of.close();

	// close input streams
	for ( int i = 0 ; i < frames.size() ; ++i ) {
		inputs[i]->close();
	}

	return EXIT_SUCCESS;
}
