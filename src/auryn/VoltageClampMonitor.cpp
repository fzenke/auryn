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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "VoltageClampMonitor.h"

using namespace auryn;

VoltageClampMonitor::VoltageClampMonitor(NeuronGroup * source, NeuronID id, std::string filename) : Monitor(filename)
{
	init(source,id,filename);
}

VoltageClampMonitor::~VoltageClampMonitor()
{
}

void VoltageClampMonitor::init(NeuronGroup * source, NeuronID id, std::string filename)
{
	// only register if the neuron exists on this rank
	src = source;
	nid = id;
	gid = src->rank2global(nid);

	clamping_voltage = -70e-3;
	clamp_enabled = true;

	t_stop = -1; // at the end of all times ...

	if ( nid < src->get_post_size() ) {
		auryn::sys->register_device(this);
		outfile << std::setiosflags(std::ios::fixed) << std::setprecision(6);
		outfile << "# Clamping neuron " << gid << "\n";
	}
}

void VoltageClampMonitor::execute()
{
	if ( clamp_enabled && auryn::sys->get_clock() < t_stop ) {
		AurynState * voltage = src->mem->ptr(nid);
		AurynState pseudo_current = clamping_voltage-*voltage;
		// TODO a unitful quantity would be nice here ... 
		*voltage += pseudo_current;
		outfile << (auryn::sys->get_time()) << " " << pseudo_current << "\n";
	}
}



void VoltageClampMonitor::record_for(AurynDouble time)
{
	set_stop_time(time);
}

void VoltageClampMonitor::set_stop_time(AurynDouble time)
{
	if (time < 0) {
		auryn::logger->msg("Warning: Negative stop times not supported -- ingoring.",WARNING);
	} 
	else t_stop = auryn::sys->get_clock() + time/auryn_timestep;
}
