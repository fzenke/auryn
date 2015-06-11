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

#include "RateChecker.h"

RateChecker::RateChecker(SpikingGroup * source, AurynFloat max) : Checker(source)
{
	init(0.,max,1.);
}

RateChecker::RateChecker(SpikingGroup * source, AurynFloat min, AurynFloat max, AurynFloat tau) : Checker(source)
{
	init(min,max,tau);
}

RateChecker::~RateChecker()
{
}

void RateChecker::init(AurynFloat min, AurynFloat max, AurynFloat tau)
{
	if ( src->evolve_locally() )
		sys->register_checker(this);
	timeconstant = tau;
	size = src->get_size();
	popmin = min;
	popmax = max;
	decay_multiplier = exp(-dt/tau);
	reset();
}


bool RateChecker::propagate()
{
	state *= decay_multiplier;
	state += 1.*src->get_spikes()->size()/timeconstant/size;
	if ( state>=popmin && state=<popmax ) return true;
	else  return false;
}

AurynFloat RateChecker::get_property()
{
	return get_rate();
}

AurynFloat RateChecker::get_rate()
{
	return state;
}

void RateChecker::reset()
{
	state = popmin;
}
