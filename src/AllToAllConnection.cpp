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

#include "AllToAllConnection.h"

using namespace auryn;

AllToAllConnection::AllToAllConnection( 
		SpikingGroup * source, 
		NeuronGroup * destination, 
		AurynWeight weight, 
		TransmitterType transmitter, 
		std::string name) 
: Connection(source,destination,transmitter,name)
{
	init(weight);
}


AllToAllConnection::~AllToAllConnection()
{
	free();
}

void AllToAllConnection::init(AurynWeight weight) 
{
	if ( dst->evolve_locally() == true )
		auryn::sys->register_connection(this);

	connection_weight = weight;
}

void AllToAllConnection::free()
{
}

void AllToAllConnection::finalize()
{
}

void AllToAllConnection::propagate()
{
	SpikeContainer::const_iterator spikes_end = src->get_spikes()->end();
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ;
			spike != spikes_end ; ++spike ) {
		target_state_vector->add( connection_weight );
	}
}

AurynWeight AllToAllConnection::get_data(NeuronID i)
{
	return 0;
}

void AllToAllConnection::set_data(NeuronID i, AurynWeight value)
{
	connection_weight = value;
}

AurynWeight AllToAllConnection::get(NeuronID i, NeuronID j)
{
	return connection_weight;
}

AurynWeight * AllToAllConnection::get_ptr(NeuronID i, NeuronID j)
{
	return &connection_weight;
}

void AllToAllConnection::set(NeuronID i, NeuronID j, AurynWeight value)
{
	connection_weight = value;
}

void AllToAllConnection::stats(AurynDouble &mean, AurynDouble &std, StateID z)
{
	mean = connection_weight;
	std = 0;
}

bool AllToAllConnection::write_to_file(std::string filename)
{
	return true; // TODO fake but what else ? 
}

bool AllToAllConnection::load_from_file(std::string filename)
{
	return true; // TODO fake but what else ? 
}

AurynLong AllToAllConnection::get_nonzero()
{
	return 1;
}


std::vector<neuron_pair> AllToAllConnection::get_block(NeuronID lo_row, NeuronID lo_col, NeuronID hi_row,  NeuronID hi_col) 
{
	std::vector<neuron_pair> clist;
	for ( NeuronID i = lo_row ; i < hi_row ; ++i ) {
		for ( NeuronID j =  lo_col ; j < hi_col ; ++j ) {
			if ( i == j ) {
				neuron_pair a;
				a.i = i;
				a.j = j;
				clist.push_back( a );
			}
		}
	}
	return clist;
}
