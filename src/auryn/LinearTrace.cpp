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

#include "LinearTrace.h"

using namespace auryn;

void LinearTrace::init(NeuronID n, AurynFloat timeconstant) 
{
	tau_auryntime = (AurynTime) (timeconstant/auryn_timestep);
	zerointerval = 5*tau_auryntime;

	// clock = auryn::sys->get_clock_ptr();
	timestamp = new AurynTime[size];
}

void LinearTrace::free()
{
	delete [] timestamp;
}

LinearTrace::LinearTrace(NeuronID n, AurynFloat timeconstant) : super(n, timeconstant)
{
	init(n,timeconstant);
	clock = sys->get_clock_ptr();
}

LinearTrace::LinearTrace(NeuronID n, AurynFloat timeconstant, AurynTime * clk ) : super(n, timeconstant)
{
	init(n,timeconstant);
	clock = clk;
}

LinearTrace::~LinearTrace()
{
	free();
}


void LinearTrace::update(NeuronID i)
{
	const int timediff = *clock - timestamp[i];
	if ( timediff == 0 ) return;

	if ( timediff >= zerointerval ) {
		data[i] = 0.0;
	} else { // as a last resort call exp
		data[i] *= std::exp( -(auryn_timestep*timediff)/tau);
	}

	timestamp[i] = *clock;
}

void LinearTrace::add_specific(NeuronID i, AurynState amount)
{
	update(i);
	super::add_specific(i, amount);
}

void LinearTrace::inc(NeuronID i)
{
	update(i);
	data[i] += 1.0;
}

AurynFloat LinearTrace::get(NeuronID i)
{
	update(i);
	return super::get(i);
}


void LinearTrace::evolve() 
{
	// lazy on evolve updates on get and inc
}
