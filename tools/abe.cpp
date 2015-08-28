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


	istream * input;

	if ( input_file_name == "-" ) {
		input = &std::cin;
	} else {
		input = new ifstream( input_file_name.c_str(), std::ios::in );
		if (!(*input)) {
			std::cerr << "Unable to open input file" << endl;
			exit(EXIT_FAILURE);
		}
	}

	// get length of the file
	input->seekg (0, input->end);
	AurynLong length = input->tellg();
	input->seekg (0, input->beg);

	// read first entry to infer dt 
	struct SpikeEvent_type spike_data;
	input->read((char*)&spike_data, sizeof(SpikeEvent_type));
	double dt = 1.0/spike_data.time;
	NeuronID group_size = spike_data.neuronID;

	cout << "Timestep: " << dt << endl;

	for (AurynLong i = 0 ; i < length ; ++i ) {
		input->read((char*)&spike_data, sizeof(SpikeEvent_type));
		cout << spike_data.time*dt << " " << spike_data.neuronID << "\n";
	}


	return EXIT_SUCCESS;
}
