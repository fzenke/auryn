/* 
* Copyright 2014-2016 Friedemann Zenke
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

#define N 1

/*!\file 
 *
 * \brief Example simulation which simulates one Poisson input with short-term plasticity onto a single 
 * postsynaptic neuron and records the membrane potential 
 *
 * As opposed to sim_epsp.cpp this examples uses a connection with short-term plasticity (Tsodyks Markram model) 
 * which is implemented in STPConnection.
 * */

using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{

	int errcode = 0;
	char strbuf [255];
	string outputfile = "out_epsp_stp";
	string tmpstr;
	AurynWeight w = 1.0;

	// BEGIN Global definitions
	auryn_init( ac, av );
	sys->set_simulation_name(outputfile);
	// END Global definitions

	// Sets up a single presynaptic Poisson neuron which fires at 1Hz
	PoissonGroup * poisson = new PoissonGroup(N,1.);

	// Sets up a single postsynaptic integrate-and-fire neuron
	IFGroup * neuron = new IFGroup(1);


	// Initializes the STP connection
	STPConnection * con = new STPConnection(poisson,neuron,w,1.0,GLUT);

	// Sets STP parameters (depression dominated)
	double U = 0.2;
	double taud = 200e-3; // s
	double tauf = 50e-3;  // s

	// Passes the parameters to the connection instance
	con->set_tau_d(taud);
	con->set_tau_f(tauf);
	con->set_ujump(U);
	con->set_urest(U);


	// Sets up recording

	// Records the input spike train
	tmpstr = outputfile;
	tmpstr += ".ras";
	SpikeMonitor * smon = new SpikeMonitor( poisson, tmpstr.c_str() );

	// Records the postsynaptic membrane potential
	tmpstr = outputfile;
	tmpstr += ".mem";
	VoltageMonitor * vmon = new VoltageMonitor( neuron, 0, tmpstr.c_str() );

	// Records the postsynaptic AMPA conductance
	tmpstr = outputfile;
	tmpstr += ".ampa";
	StateMonitor * amon = new StateMonitor( neuron, 0, "g_ampa", tmpstr.c_str() );

	// Records the postsynaptic NMDA conductance
	tmpstr = outputfile;
	tmpstr += ".nmda";
	StateMonitor * nmon = new StateMonitor( neuron, 0, "g_nmda", tmpstr.c_str() );

	// simulate for 5s
	logger->msg("Running ...",PROGRESS);
	sys->run(5);

	// Changes firing rate of PoissonGroup
	poisson->set_rate(50.0);

	// simulate for 0.5s
	sys->run(0.5);

	// Changes firing rate of PoissonGroup
	poisson->set_rate(1.0);

	// simulate for 4.5s
	sys->run(4.5);


	if (errcode)
		auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();

	return errcode;
}
