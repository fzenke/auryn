/* 
* Copyright 2014-2018 Friedemann Zenke
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
 * \brief A minimal example for simulating a balanced network model
 *
 * This is the code for the Tutorial 2. The tutorials can be found at 
 * http://www.fzenke.net/auryn/doku.php?id=tutorials:start
 *
 * */

#include "auryn.h"

using namespace auryn;

int main(int ac, char* av[]) 
{
	// Initialize Auryn kernel
	auryn_init( ac, av );

	// Set master random seed
	sys->set_master_seed(42);

	// Initialize excitatory and inhibitory network neurons
	int nb_exc_neurons = 20000;
	int nb_inh_neurons = nb_exc_neurons/4;
	IFGroup * neurons_exc = new IFGroup(nb_exc_neurons);
	neurons_exc->set_name("exc neurons");
	neurons_exc->get_state_vector("g_nmda")->set_random();
	IFGroup * neurons_inh = new IFGroup(nb_inh_neurons);
	neurons_inh->set_tau_mem(5e-3);
	neurons_inh->set_name("inh neurons");

	// Initialize Poisson input population (PoissonGroup)
	int nb_input_neurons = 5000;
	float poisson_rate = 2.0;
	PoissonGroup * poisson = new PoissonGroup(nb_input_neurons,poisson_rate);

	// Connect Poisson input to exc population
	float weight = 0.2; // conductance amplitude in units of leak conductance
	float sparseness = 0.05; // probability of connection
	SparseConnection * con_ext_exc = new SparseConnection(poisson,neurons_exc,weight,sparseness,GLUT);

	// Set up recurrent connections
	float gamma = 4.0;
	SparseConnection * con_ee = new SparseConnection(neurons_exc,neurons_exc,weight,sparseness,GLUT);
	SparseConnection * con_ei = new SparseConnection(neurons_exc,neurons_inh,weight,sparseness,GLUT);
	SparseConnection * con_ie = new SparseConnection(neurons_inh,neurons_exc,gamma*weight,sparseness,GABA);
	SparseConnection * con_ii = new SparseConnection(neurons_inh,neurons_inh,gamma*weight,sparseness,GABA);

	// Initialize monitors which record information to file
	
	// Set up to display the population firing rate in the progress bar
	sys->set_online_rate_monitor_id(neurons_exc->get_uid());
	
	// Record input and output spikes
	SpikeMonitor * exc_spike_mon = new SpikeMonitor( neurons_exc, sys->fn("exc","ras") );

	// Record membrane voltage of one of the neurons
	VoltageMonitor * voltage_mon = new VoltageMonitor( neurons_exc, 0, sys->fn("neuron","mem") );
	voltage_mon->record_for(2); // only record for 2 seconds

	// Run the simulation for 10 seconds
	sys->run(10);

	// let's add a cell assembly 
	con_ee->set_block(0,500,0,500,5*weight);

	// Run the simulation for another 2 seconds
	sys->run(2);

	// Shut down Auryn kernel
	auryn_free();
}
