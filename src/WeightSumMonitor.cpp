/* 
* Copyright 2014 Friedemann Zenke
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

#include "WeightSumMonitor.h"

WeightSumMonitor::WeightSumMonitor(Connection * source, string filename, AurynDouble binsize) : Monitor(filename)
{
	init(source,filename,binsize/dt);
}

WeightSumMonitor::~WeightSumMonitor()
{
}

void WeightSumMonitor::init(Connection * source, string filename,AurynTime stepsize)
{
	if ( !source->get_destination()->evolve_locally() ) return;

	sys->register_monitor(this);

	src = source;
	ssize = stepsize;
	outfile << setiosflags(ios::fixed) << setprecision(6);
}

void WeightSumMonitor::propagate()
{
	if (sys->get_clock()%ssize==0) {
		AurynFloat mean,std;
		src->stats(mean,std);
		outfile << (sys->get_time()) << " " << mean << " "  << std << endl;
	}

}
