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

/*! Auryn Binary Extract ABE extracts spikes or timeseries data from
 * binary files written by BinarySpikeMonitor or BinaryStateMonitor. */

#include "auryn.h"
#include <iostream>
#include <fstream>

using namespace std;

namespace po = boost::program_options;

SpikeEvent_type get( AurynLong index, ifstream * input ) 
{
	input->seekg (0, input->beg); 
	SpikeEvent_type tmp;
	for (AurynLong i = 0 ; i < index ; ++i ) {
		input->read((char*)&tmp, sizeof(SpikeEvent_type));
	}
	return tmp;
}


/*! Finds element index associated with timestamp
 * given. Return first or last element if the 
 * timestamp is out of range. */
AurynLong find( AurynDouble timestamp, ifstream * file )
{
	long unsigned int lo = 0;
	file->seekg (0, file->end);
	long unsigned int hi = file->tellg()/sizeof(SpikeEvent_type);

	while ( (lo+1) < hi ) {	
		long unsigned int pivot = lo + ( hi + lo )/2;

		SpikeEvent_type spike_data = get( pivot, file);
		cout << lo << " " << hi << " " 
			<< pivot << " " << spike_data.time << endl;

		if ( timestamp >= spike_data.time ) {
			lo = pivot;
		} else {
			hi = pivot;
		}
		
	}

	return hi;
}


int main(int ac, char* av[]) 
{
	string input_file_name = "-";

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("file", po::value<string>(), "input file")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        if (vm.count("file")) {
			input_file_name = vm["file"].as<string>();
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
	input = new ifstream( input_file_name.c_str(), std::ios::in );
	if (!(*input)) {
		std::cerr << "Unable to open input file" << endl;
		exit(EXIT_FAILURE);
	}

	// get length of the file
	input->seekg (0, input->end);
	AurynLong length = input->tellg();
	input->seekg (0, input->beg); 

	// read first entry to infer dt 
	struct SpikeEvent_type spike_data;
	input->read((char*)&spike_data, sizeof(SpikeEvent_type));
	double dt = 1.0/spike_data.time;
	dt = 1e-4; // FIXME
	NeuronID group_size = spike_data.neuronID;

	cout << "Timestep: " << dt << endl;
	cout << "Length: " << length << endl;
	cout << "Sizeof: " << sizeof(SpikeEvent_type) << endl;
	AurynLong num_of_entries = length/sizeof(SpikeEvent_type);

	int res = find(0.5, input);
	SpikeEvent_type tmp = get( res, input );
	cout << "index " << res << " time " << tmp.time << endl;


	// for (AurynLong i = res ; i < num_of_entries ; ++i ) {
	// 	input->read((char*)&spike_data, sizeof(SpikeEvent_type));
	// 	cout << spike_data.time << " " << spike_data.neuronID << "\n"; // FIXME
	// }


	return EXIT_SUCCESS;
}
