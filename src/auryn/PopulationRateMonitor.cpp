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

#include "PopulationRateMonitor.h"

using namespace auryn;

PopulationRateMonitor::PopulationRateMonitor(SpikingGroup * source, std::string filename, AurynDouble binsize) : Monitor(filename)
{
	init(source,filename,binsize);
}

PopulationRateMonitor::~PopulationRateMonitor()
{
}

void PopulationRateMonitor::init(SpikingGroup * source, std::string filename, AurynDouble binsize)
{
	auryn::sys->register_device(this);

	src = source;
	invbsize = 1.0/binsize;
	ssize = (1.0*binsize/auryn_timestep);
	if ( ssize < 1 ) ssize = 1;
	counter = 0;

	std::stringstream oss;
	oss << "PopulationRateMonitor:: Setting binsize " << binsize << "s";
	auryn::logger->msg(oss.str(),NOTIFICATION);

	// outfile << std::setiosflags(ios::fixed) << setprecision(6);
}

void PopulationRateMonitor::execute()
{
	if ( src->evolve_locally() ) {
		counter += src->get_spikes_immediate()->size();
		if (auryn::sys->get_clock()%ssize==0) {
			double rate = invbsize*counter/src->get_rank_size();
			counter = 0;
			outfile << auryn::sys->get_time() << " " << rate << "\n";
		}
	}
}

void PopulationRateMonitor::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	ar & counter ;
}

void PopulationRateMonitor::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	ar & counter ;
}
