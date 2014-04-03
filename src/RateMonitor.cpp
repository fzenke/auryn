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

#include "RateMonitor.h"

RateMonitor::RateMonitor(SpikingGroup * source, string filename, AurynFloat samplinginterval) : Monitor(filename)
{
	init(source,filename,samplinginterval);
}

RateMonitor::~RateMonitor()
{
}

void RateMonitor::init(SpikingGroup * source, string filename, AurynFloat samplinginterval)
{
	sys->register_monitor(this);

	src = source;
	ssize = samplinginterval/dt;
	tau_filter = 3*samplinginterval;

	tr_post = source->get_post_trace(tau_filter);

	stringstream oss;
	oss << "RateMonitor:: Setting sampling interval " << samplinginterval << "s";
	logger->msg(oss.str(),NOTIFICATION);

	outfile << setiosflags(ios::fixed) << setprecision(6);
}

void RateMonitor::propagate()
{
	if ( src->evolve_locally() ) {
		if (sys->get_clock()%ssize==0) {
			outfile << dt*(sys->get_clock()) << " "; 
			for  (NeuronID i = 0 ; i < src->get_rank_size() ; ++i ) {
				outfile << tr_post->normalized_get(i)
					<< " "; 
			}
			outfile << "\n";
		}
	}
}
