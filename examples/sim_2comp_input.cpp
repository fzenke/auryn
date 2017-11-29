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

#include "auryn.h"

using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{
	int errcode = 0;
	char strbuf [255];
	string simname = "nbg";
	auryn_init( ac, av, ".", simname );

	// define neuron group
	NaudGroup * neurons_exc = new NaudGroup(40);
	IFGroup * neurons_inh = new IFGroup(10);

	// define input group
	PoissonGroup * poisson_exc = new PoissonGroup(40, 5);

	// connect
	const double we = 0.5;
	const double wi = 1.0;
	const double sparseness = 0.5;
	SparseConnection * con_ext_exc = new SparseConnection(poisson_exc, neurons_exc, we, sparseness);
	con_ext_exc->set_target("g_ampa_dend");
	SparseConnection * con_exc_inh = new SparseConnection(neurons_exc, neurons_inh, we, sparseness);
	SparseConnection * con_inh_exc = new SparseConnection(neurons_inh, neurons_exc, wi, sparseness, GABA);
	con_inh_exc->set_target("g_gaba_dend");

	// define monitors
	VoltageMonitor * vmon   = new VoltageMonitor( neurons_exc, 0, sys->fn("mem") );
	SpikeMonitor * smon = new SpikeMonitor( neurons_exc, sys->fn("ras") );
	StateMonitor * smon_vd  = new StateMonitor( neurons_exc, 0, "Vd", sys->fn("Vd") );
	StateMonitor * smon_m   = new StateMonitor( neurons_exc, 0, "thr", sys->fn("thr") );
	StateMonitor * smon_ws  = new StateMonitor( neurons_exc, 0, "wsoma", sys->fn("wsoma") );
	StateMonitor * smon_wd  = new StateMonitor( neurons_exc, 0, "wdend", sys->fn("wdend") );

	// run simulation
	logger->msg("Running ...",PROGRESS);

	const double simtime = 2;
	// simulate
	sys->run(simtime);

	if (errcode)
		auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
	return errcode;
}
