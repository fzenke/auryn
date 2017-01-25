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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "DelayedSpikeMonitor.h"

using namespace auryn;

DelayedSpikeMonitor::DelayedSpikeMonitor(SpikingGroup * source, std::string filename, NeuronID from, NeuronID to) 
	: Monitor(filename)
{
	init(source,filename,from,to);
}

DelayedSpikeMonitor::DelayedSpikeMonitor(SpikingGroup * source, std::string filename, NeuronID to)
	: Monitor(filename)
{
	init(source,filename,0,to);
}

DelayedSpikeMonitor::DelayedSpikeMonitor(SpikingGroup * source, std::string filename)
	: Monitor(filename)
{
	init(source,filename,0,source->get_size());
}

DelayedSpikeMonitor::~DelayedSpikeMonitor()
{
	free();
}

void DelayedSpikeMonitor::init(SpikingGroup * source, std::string filename, NeuronID from, NeuronID to)
{
	auryn::sys->register_device(this);

	// sys = system;
	n_from = from;
	n_to = to;
	src = source;
	offset = 0;
	outfile.setf(std::ios::fixed);
	outfile.precision(5); 
}

void DelayedSpikeMonitor::free()
{
}

void DelayedSpikeMonitor::set_offset(NeuronID of)
{
	offset = of;
}

void DelayedSpikeMonitor::execute()
{
	for (it = src->get_spikes()->begin() ; it != src->get_spikes()->end() ; ++it ) {
		if (*it >= n_from ) {
			if ( *it < n_to ) 
			 outfile << auryn_timestep*(auryn::sys->get_clock()) << "  " << *it+offset << "\n";
		}
	}
}
