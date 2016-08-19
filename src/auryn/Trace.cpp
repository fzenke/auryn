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

#include "Trace.h"

using namespace auryn;



Trace::Trace(NeuronID n, AurynFloat timeconstant) : AurynStateVector(n)
{
	set_timeconstant(timeconstant);
}

Trace::~Trace()
{
}

void Trace::set_timeconstant(AurynFloat timeconstant)
{
	tau = timeconstant;
}

AurynFloat Trace::get_tau()
{
	return tau;
}

AurynStateVector * Trace::get_state_ptr()
{
	return this;
}

void Trace::inc(NeuronID i)
{
   data[i]++;
}

void Trace::inc(SpikeContainer * sc)
{
	for ( NeuronID i = 0 ; i < sc->size() ; ++i )
		inc((*sc)[i]);
}

AurynFloat Trace::normalized_get(NeuronID i)
{
	return get( i ) / tau ;
}

