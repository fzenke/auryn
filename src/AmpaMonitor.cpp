/* 
* Copyright 2014-2015 Friedemann Zenke
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

#include "AmpaMonitor.h"

AmpaMonitor::AmpaMonitor(NeuronGroup * source, NeuronID id, string filename, AurynTime stepsize) : Monitor(filename)
{
	init(source,id,filename,stepsize);
}

AmpaMonitor::~AmpaMonitor()
{
}

void AmpaMonitor::init(NeuronGroup * source, NeuronID id, string filename, AurynTime stepsize)
{
	sys->register_monitor(this);

	src = source;
	ssize = stepsize;
	if ( ssize < 1 ) ssize = 1;

	nid = id;
	outfile << setiosflags(ios::fixed) << setprecision(6);
}

void AmpaMonitor::propagate()
{
	if (sys->get_clock()%ssize==0) {
		outfile << dt*(sys->get_clock()) << " " << src->get_ampa(nid) << "\n";
	}

}
