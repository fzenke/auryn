/* 
* Copyright 2014-2017 Friedemann Zenke
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

/*!\file 
 *
 * \brief A minimal example for simulating a single AdEx neuron with excitatory Poisson input
 *
 * This is the code for the Tutorial 1. The tutorials can be found at 
 * http://www.fzenke.net/auryn/doku.php?id=tutorials:tutorial_1
 *
 * */

#include "auryn.h"

using namespace auryn;

int main(int ac, char* av[]) 
{
	// Initialize Auryn kernel
	auryn_init( ac, av );

	// Set master random seed
	sys->set_master_seed(102);

	// Initialize Poisson input population (PoissonGroup)
	int nb_input_neurons = 100;
	float poisson_rate = 5.0;
	PoissonGroup * poisson = new PoissonGroup(nb_input_neurons,poisson_rate);

	// Initialize receiving population with one neuron
	AdExGroup * neuron = new AdExGroup(1);

	// Connect input and the AdEx neuron
	float weight = 0.2; // conductance amplitude in units of leak conductance
	AllToAllConnection * con = new AllToAllConnection(poisson,neuron,weight);
	con->set_transmitter(GLUT); // Make this an excitatory connection

	// Initialize monitors which record information to file
	//
	// Record input and output spikes
	SpikeMonitor * input_spike_mon = new SpikeMonitor( poisson, sys->fn("input","ras") );
	SpikeMonitor * output_spike_mon = new SpikeMonitor( neuron, sys->fn("output","ras") );

	// Record membrane voltage of AdEx neuron
	VoltageMonitor * output_voltage_mon = new VoltageMonitor( neuron, 0, sys->fn("output","mem") );

	// Run the simulation for 2 seconds
	sys->run(2);

	// Shut down Auryn kernel
	auryn_free();
}
