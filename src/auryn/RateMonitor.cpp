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

#include "RateMonitor.h"

using namespace auryn;

RateMonitor::RateMonitor(SpikingGroup * source, std::string filename, AurynFloat samplinginterval) : Monitor(filename)
{
	init(source,filename,samplinginterval);
}

RateMonitor::~RateMonitor()
{
}

void RateMonitor::init(SpikingGroup * source, std::string filename, AurynFloat samplinginterval)
{
	auryn::sys->register_device(this);

	src = source;
	ssize = samplinginterval/auryn_timestep;
	if ( ssize < 1 ) ssize = 1;

	tau_filter = 3*samplinginterval;

	tr_post = source->get_post_trace(tau_filter);

	std::stringstream oss;
	oss << "RateMonitor:: Setting sampling interval " << samplinginterval << "s";
	auryn::logger->msg(oss.str(),NOTIFICATION);

	outfile << std::setiosflags(std::ios::fixed) << std::setprecision(6);
}

void RateMonitor::execute()
{
	if ( src->evolve_locally() ) {
		if (auryn::sys->get_clock()%ssize==0) {
			outfile << auryn_timestep*(auryn::sys->get_clock()) << " "; 
			for  (NeuronID i = 0 ; i < src->get_rank_size() ; ++i ) {
				outfile << tr_post->normalized_get(i)
					<< " "; 
			}
			outfile << "\n";
		}
	}
}
