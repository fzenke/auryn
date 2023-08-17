/* 
* Copyright 2014-2023 Friedemann Zenke
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

#include "DelayConnection.h"

using namespace auryn;


DelayConnection::DelayConnection(NeuronID rows, NeuronID cols) 
: SparseConnection(rows, cols)
{
	init();
}

DelayConnection::DelayConnection( SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		TransmitterType transmitter, std::string name) 
: SparseConnection(source,destination,weight,sparseness,transmitter, name)
{
	init();
}


DelayConnection::~DelayConnection()
{
	free();
}


void DelayConnection::init() 
{
	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

	src_dly = new SpikeDelay( 1 );
	src_dly->set_clock_ptr( sys->get_clock_ptr() ); // assign clock for ring buffer
}


void DelayConnection::free()
{
	delete src_dly;
}


void DelayConnection::propagate()
{
	// pop buffer from delay and push spikes from src to back
	src_dly->get_spikes_immediate()->clear();
	src_dly->push_back(src->get_spikes());


	// do normal forward propagation as in SparseConnection
	for (SpikeContainer::const_iterator spike = src_dly->get_spikes()->begin() ;
			spike != src_dly->get_spikes()->end() ; 
			++spike ) {
		for ( AurynLong c = w->get_row_begin_index(*spike) ;
				c < w->get_row_end_index(*spike) ;
				++c ) {
			transmit( w->get_colind(c) , w->get_value(c) );
		}
	}
}


void DelayConnection::set_delay_steps(unsigned int delay)
{
	delay = std::max((unsigned int)1,delay);
	src_dly->set_delay(delay);
}


void DelayConnection::set_delay(double delay)
{
	set_delay_steps( delay/auryn_timestep );
}
