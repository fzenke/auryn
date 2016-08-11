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

#include "IdentityConnection.h"

using namespace auryn;

IdentityConnection::IdentityConnection( SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, 
		TransmitterType transmitter, std::string name) 
: Connection(source,destination,transmitter,name)
{
	init(weight);
}


IdentityConnection::IdentityConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, 
		NeuronID lo_row, NeuronID hi_row, 
		NeuronID lo_col, NeuronID hi_col, 
		TransmitterType transmitter, 
		std::string name) 
: Connection(source,destination,transmitter)
{
	init(weight);
	lo_src = lo_row;
	hi_src = hi_row;
	lo_dst = lo_dst;
	hi_dst = hi_dst;
}

IdentityConnection::~IdentityConnection()
{
	free();
}

void IdentityConnection::init(AurynWeight weight) 
{
	if ( dst->evolve_locally() == true )
		auryn::sys->register_connection(this);
	if ( src == dst ) {
		std::stringstream oss;
		oss << "IdentityConnection: ("<< get_name() <<"): Detected recurrent connection. This seemingly does not make any sense!";
		auryn::logger->msg(oss.str(),WARNING);
	}
	connection_weight = weight;
	offset = 0 ;
	every = 1;
}

void IdentityConnection::free()
{
}

void IdentityConnection::finalize()
{
}

void IdentityConnection::propagate()
{
	SpikeContainer::const_iterator spikes_end = src->get_spikes()->end();
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ;
			spike != spikes_end ; ++spike ) {
		if ( *spike%every == 0 )
			safe_transmit( ( *spike/every ) + offset , connection_weight ); 
		// IMPORTANT the use of safe_transmit
		// is important here since there is 
		// no weight matrix with only the 
		// corresponding columns
	}
}

AurynWeight IdentityConnection::get_data(NeuronID i)
{
	return 0;
}

void IdentityConnection::set_data(NeuronID i, AurynWeight value)
{
}

AurynWeight IdentityConnection::get(NeuronID i, NeuronID j)
{
	if ( i == j ) return connection_weight;
	else return 0;
}

AurynWeight * IdentityConnection::get_ptr(NeuronID i, NeuronID j)
{
	if ( i == j ) return &connection_weight;
	else return NULL;
}

void IdentityConnection::set(NeuronID i, NeuronID j, AurynWeight value)
{
}

AurynLong IdentityConnection::get_nonzero()
{
	return std::min(src->get_pre_size(),dst->get_post_size());
}

void IdentityConnection::stats(AurynDouble &mean, AurynDouble &std, StateID z)
{
	mean = connection_weight;
	std = 0;
}

AurynDouble IdentityConnection::sum()
{
	return connection_weight*get_nonzero();
}

bool IdentityConnection::write_to_file(std::string filename)
{
	return true; // TODO fake but what else ? 
}

bool IdentityConnection::load_from_file(std::string filename)
{
	return true; // TODO fake but what else ? 
}


std::vector<neuron_pair> IdentityConnection::get_block(NeuronID lo_row, NeuronID lo_col, NeuronID hi_row,  NeuronID hi_col) 
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

AurynFloat IdentityConnection::mean() 
{
	AurynDouble m,s;
	stats(m,s);
	return m;
}

void IdentityConnection::set_offset(int off)
{
	offset = off;
	auryn::logger->parameter("offset",(int)offset);
}

void IdentityConnection::set_every(NeuronID e)
{
	if ( e != 0 ) {
		every = e;
		auryn::logger->parameter("every",(int)every);
	}
}

