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

#include "WeightSumMonitor.h"

using namespace auryn;

WeightSumMonitor::WeightSumMonitor(Connection * source, std::string filename, AurynDouble binsize) : Monitor(filename)
{
	init(source,filename,binsize/auryn_timestep);
}

WeightSumMonitor::~WeightSumMonitor()
{
}

void WeightSumMonitor::init(Connection * source, std::string filename,AurynTime stepsize)
{
	if ( !source->get_destination()->evolve_locally() ) return;

	auryn::sys->register_device(this);

	src = source;
	ssize = stepsize;
	if ( ssize < 1 ) ssize = 1;

	outfile << std::setiosflags(std::ios::fixed) << std::setprecision(6);
}

void WeightSumMonitor::execute()
{
	if (auryn::sys->get_clock()%ssize==0) {
		AurynDouble weightsum;
		AurynDouble weightstd;
		src->stats(weightsum, weightstd);
		outfile << (auryn::sys->get_time()) << " " << weightsum << std::endl;
	}

}
