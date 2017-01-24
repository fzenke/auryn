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

#include "WeightMatrixMonitor.h"

using namespace auryn;

WeightMatrixMonitor::WeightMatrixMonitor(Connection * source, std::string filename, AurynFloat stepsize) : Monitor(filename)
{
	init(source,stepsize);
}

WeightMatrixMonitor::~WeightMatrixMonitor()
{
}

void WeightMatrixMonitor::init(Connection * source, AurynFloat stepsize)
{
	auryn::sys->register_device(this);

	src = source;
	ssize = (AurynTime) (stepsize/auryn_timestep);
	outfile << std::setiosflags(std::ios::fixed) << std::setprecision(6);

	filecount = 0;
}

void WeightMatrixMonitor::execute()
{
	if (auryn::sys->get_clock()%ssize==0) {
		AurynDouble mean,std;
		src->stats(mean,std);

		char wmatfilename [255];
		sprintf(wmatfilename, "%s%.2d", fname.c_str(), filecount);
		src->write_to_file(wmatfilename);
		filecount++;

		outfile << auryn_timestep*(auryn::sys->get_clock()) << " " << mean << " "  << std << " " << wmatfilename << std::endl;
	}

}
