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
*/

#include "auryn.h"

#define N 1

/*!\file 
 *
 * \brief Example simulation illustrating the effect of delay connection.
 *
 * The simulation sets up a single input neuron and connects it to a single output neuron via a SparseConnection
 * (which only has the axonal delay from IFGroup) and via a DelayConnection which has both the axonal delay plus
 * an added 10ms dendritic delay. The latter connection is GABAergic to make it easily differntiable in the 
 * membrane trace.
 * */

using namespace auryn;

int main(int ac, char* av[]) 
{

	int errcode = 0;
	string simname = "delay_connection";
	string logfile = simname;
	string tmpstr;
	AurynWeight w = 1.0;

	// BEGIN Global definitions
	auryn_init( ac, av );
	sys->set_simulation_name(simname);
	// END Global definitions

	// define input group
	PoissonGroup * poisson = new PoissonGroup(N,1.);

	// define receiving group
	IFGroup * neuron = new IFGroup(1);

	// define connections
	SparseConnection * con = new SparseConnection(poisson, neuron, w, 1.0, GLUT);
	DelayConnection * dly_con = new DelayConnection(poisson, neuron, w,  1.0, GABA);
	dly_con->set_delay(10e-3); // 10ms delay added

	// define monitors
	SpikeMonitor * smon = new SpikeMonitor( neuron, sys->fn("ras") );
	VoltageMonitor * vmon = new VoltageMonitor( neuron, 0, sys->fn("mem"), 1e-3 );
	StateMonitor * amon = new StateMonitor( neuron, 0, "g_ampa", sys->fn("ampa") );
	StateMonitor * nmon = new StateMonitor( neuron, 0, "g_gaba", sys->fn("gaba") );

	// run simulation
	logger->msg("Running ...",PROGRESS);
	sys->run(10);

	if (errcode)
		auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
	return errcode;
}
