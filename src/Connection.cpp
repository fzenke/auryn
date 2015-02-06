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

#include "Connection.h"

Connection::Connection()
{
	init();
}

Connection::Connection(NeuronID rows, NeuronID cols)
{
	init();
	set_size(rows,cols);
}

Connection::Connection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter, string name)
{
	set_source(source);
	set_destination(destination);
	m_rows = src->get_size();
	n_cols = dst->get_size();
	init(transmitter);
	set_name(name);
}

void Connection::init(TransmitterType transmitter)
{
	set_transmitter(transmitter);
	set_name("Unspecified");
}

void Connection::set_size(NeuronID i, NeuronID j)
{
	m_rows = i;
	n_cols = j;
}


Connection::~Connection()
{
}

void Connection::set_name(string name)
{
	connection_name = name;
}

string Connection::get_name()
{
	return connection_name;
}

TransmitterType Connection::get_transmitter()
{
	return trans;
}

void Connection::set_transmitter(TransmitterType transmitter)
{
	trans = transmitter;
	if ( dst->evolve_locally() ) {
		switch ( transmitter ) {
			case GABA:
				set_transmitter(dst->get_gaba_ptr()->data);
				break;
			case MEM:
				set_transmitter(dst->get_mem_ptr()->data);
				break;
			case NMDA:
				set_transmitter(dst->get_nmda_ptr()->data);
				break;
			case GLUT:
			case AMPA:
			default:
				set_transmitter(dst->get_ampa_ptr()->data);
		}
	} else set_transmitter(NULL);
}

void Connection::set_transmitter(AurynWeight * ptr)
{
	target = ptr;
}

NeuronID Connection::get_m_rows()
{
	return m_rows;
}

NeuronID Connection::get_n_cols()
{
	return n_cols;
}

void Connection::set_source(SpikingGroup * source)
{
	src = source;
}

SpikingGroup * Connection::get_source()
{
	return src ;
}

void Connection::set_destination(NeuronGroup * destination)
{
	dst = destination;
}

NeuronGroup *  Connection::get_destination()
{
	return dst;
}

void Connection::conditional_propagate() 
{
	if ( dst->evolve_locally() ) 
		propagate();
}


void Connection::safe_transmit(NeuronID id, AurynWeight amount) 
{
	if ( dst->localrank(id) )
		transmit( id, amount );
}

void Connection::evolve() 
{

}
