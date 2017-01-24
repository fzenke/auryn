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

#include "SpikeMonitor.h"

using namespace auryn;

SpikeMonitor::SpikeMonitor(SpikingGroup * source, std::string filename, NeuronID from, NeuronID to) 
	: Monitor(filename)
{
	init(source,filename,from,to);
}

SpikeMonitor::SpikeMonitor(SpikingGroup * source, std::string filename, NeuronID to)
	: Monitor(filename)
{
	init(source,filename,0,to);
}

SpikeMonitor::SpikeMonitor(SpikingGroup * source, std::string filename)
	: Monitor(filename)
{
	init(source,filename,0,source->get_size());
}

SpikeMonitor::~SpikeMonitor()
{
	free();
}

void SpikeMonitor::init(SpikingGroup * source, std::string filename, NeuronID from, NeuronID to)
{
	auryn::sys->register_device(this);

	// sys = system;
	active = true;
	n_from = from;
	n_to = to;
	n_every = 1;
	src = source;
	outfile.setf(std::ios::fixed);
	outfile.precision(log(auryn_timestep)/log(10)+1 );
}

void SpikeMonitor::free()
{
}


void SpikeMonitor::set_every(NeuronID every)
{
	n_every = every;
}

void SpikeMonitor::execute()
{
	if ( !active ) return;

	for (it = src->get_spikes_immediate()->begin() ; it < src->get_spikes_immediate()->end() ; ++it ) {
		if (*it >= n_from ) {
			if ( *it < n_to && (*it%n_every==0) ) {
			  // using the good old stdio.h for formatting seems a bit faster 
			  char buffer[255];
			  int n = sprintf(buffer,"%f %u\n",auryn::sys->get_time(), *it); 
			  outfile.write(buffer,n); 
			}
		}
	}
}
