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

#include "PopulationRateMonitor.h"

PopulationRateMonitor::PopulationRateMonitor(SpikingGroup * source, string filename, AurynFloat binsize) : Monitor(filename)
{
	init(source,filename,binsize);
}

PopulationRateMonitor::~PopulationRateMonitor()
{
}

void PopulationRateMonitor::init(SpikingGroup * source, string filename, AurynFloat binsize)
{
	sys->register_monitor(this);

	src = source;
	bsize = binsize;
	ssize = bsize/dt;
	counter = 0;

	stringstream oss;
	oss << "PopulationRateMonitor:: Setting binsize " << bsize << "s";
	logger->msg(oss.str(),NOTIFICATION);

	outfile << setiosflags(ios::fixed) << setprecision(6);
}

void PopulationRateMonitor::propagate()
{
	if ( src->evolve_locally() ) {
		counter += src->get_spikes_immediate()->size();
		if (sys->get_clock()%ssize==0) {
			double rate = 1.*counter/bsize/src->get_rank_size();
			counter = 0;
			outfile << dt*(sys->get_clock()) << " " << rate << "\n";
		}
	}
}
