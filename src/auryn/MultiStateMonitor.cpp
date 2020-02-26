/* 
* Copyright 2014-2020 Friedemann Zenke
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

#include "MultiStateMonitor.h"

using namespace auryn;

MultiStateMonitor::MultiStateMonitor(std::string filename, AurynDouble stepsize) : StateMonitor(filename, stepsize)
{
}

MultiStateMonitor::~MultiStateMonitor()
{
}


void MultiStateMonitor::add_target(AurynState * target)
{
	state_list.push_back(target);
}

void MultiStateMonitor::add_neuron_range(AurynStateVector * source, NeuronID from, NeuronID to)
{
	for ( unsigned int i = from ; i < to ; ++i ) {
		add_target(source->ptr(i));
	}
}

void MultiStateMonitor::add_neuron_range(AurynStateVector * source)
{
	add_neuron_range(source,0,source->size);
}

void MultiStateMonitor::add_neuron_range(NeuronGroup * source, std::string statename)
{
	add_neuron_range(source->get_state_vector(statename));
}

void MultiStateMonitor::add_neuron_range(NeuronGroup * source, std::string statename, NeuronID from, NeuronID to)
{
	add_neuron_range(source->get_state_vector(statename), from, to);
}


void MultiStateMonitor::execute()
{
	if ( auryn::sys->get_clock() < t_stop ) {
		if ( (auryn::sys->get_clock())%ssize==0 ) {
			outfile << (auryn::sys->get_time()) << " ";
			for (std::vector<AurynState*>::const_iterator iter = state_list.begin() ; 
			     iter != state_list.end();
				 ++iter ) {
				outfile << **iter << " ";
			}
			outfile << "\n";
		}
	}
}


