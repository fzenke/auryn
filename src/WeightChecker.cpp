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

#include "WeightChecker.h"

using namespace auryn;

WeightChecker::WeightChecker(Connection * source, AurynFloat max) : Checker()
{
	init(source,0.,max,10.);
}

WeightChecker::WeightChecker(Connection * source, AurynFloat min, AurynFloat max, AurynFloat timestep) : Checker()
{
	init(source,min,max,timestep);
}

WeightChecker::~WeightChecker()
{
}

void WeightChecker::init(Connection * source, AurynFloat min, AurynFloat max, AurynFloat timestep)
{
	auryn::sys->register_checker(this);
	logger->msg("WeightChecker:: Initializing", VERBOSE);

	source_ = source;
	wmin = min;
	wmax = max;

	if (timestep<0.0) {
		logger->msg("WeightChecker:: Minimally allowed timestep is 1dt", WARNING);
		timestep = 1;
	} else timestep_ = timestep/dt;

}


bool WeightChecker::propagate()
{

	if ( (sys->get_clock()%timestep_) == 0 ) {
		AurynDouble mean, std;
		source_->stats(mean, std);
		if ( mean<wmin || mean>wmax ) { 
			std::stringstream oss;
			oss << "WeightChecker:: Detected mean weight of " << mean ;
			logger->msg(oss.str(),WARNING);
			return false; // break run
		}
	}

	return true; 
}

