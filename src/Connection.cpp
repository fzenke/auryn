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

#include "Connection.h"

using namespace auryn;

Connection::Connection()
{
	init();
}

Connection::Connection(NeuronID rows, NeuronID cols)
{
	init();
	set_size(rows,cols);
}

Connection::Connection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter, std::string name)
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

	number_of_spike_attributes = 0;

	// Here we store how many spike attributes have already been 
	// added to the stack due to other connections having the same
	// source SpikingGroup. 
	spike_attribute_offset = src->get_num_spike_attributes();
}

void Connection::set_size(NeuronID i, NeuronID j)
{
	m_rows = i;
	n_cols = j;
}


Connection::~Connection()
{
}

void Connection::set_name(std::string name)
{
	connection_name = name;
}

std::string Connection::get_name()
{
	return connection_name;
}

std::string Connection::get_file_name()
{
	std::string filename (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__);
	return filename;
}

std::string Connection::get_log_name()
{
	std::stringstream oss;
	oss << get_name() << " ("
		<< get_file_name() << "): ";
	return oss.str();
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
				set_target(dst->get_state_vector("g_gaba"));
				break;
			case MEM:
				set_target(dst->get_state_vector("mem"));
				break;
			case CURSYN:
				set_target(dst->get_state_vector("g_cursyn"));
				break;
			case NMDA:
				set_target(dst->get_state_vector("g_nmda"));
				break;
			case GLUT:
			case AMPA:
			default:
				set_target(dst->get_state_vector("g_ampa"));
		}
	} else set_target((AurynWeight *)NULL);
}


void Connection::set_target(AurynWeight * ptr)
{
	target = ptr;
}

void Connection::set_target(AurynStateVector * ptr)
{
	target_state_vector = ptr;
	set_target(ptr->data);
}

void Connection::set_receptor(AurynStateVector * ptr)
{
	set_target(ptr);
}

void Connection::set_receptor(string state_name)
{
	set_receptor(dst->get_state_vector(state_name));
}

void Connection::set_target(string state_name)
{
	set_receptor(state_name);
}


void Connection::set_transmitter(string state_name)
{
	set_receptor(state_name);
}

AurynStateVector * Connection::get_target_vector()
{
	return target_state_vector;
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

void Connection::add_number_of_spike_attributes(int x)
{
	if ( x <= 0 ) {
		throw AurynSpikeAttributeSizeException();
	}

	number_of_spike_attributes += x; // we remember how many attributes are due to this connection
	src->inc_num_spike_attributes(x);
}

SpikeContainer * Connection::get_pre_spikes()
{
	return src->get_spikes();
}

SpikeContainer * Connection::get_post_spikes()
{
	return dst->get_spikes_immediate();
}


AurynFloat Connection::get_spike_attribute(const NeuronID spike_array_pos, const int attribute_id)
{
	// We need to skip attributes by other Connection objects (spike_attribute_offset)
	// and other attributes from this Connection. Note that if attribute_id is larger
	// then number_of_spike_attributes the behavior will be undefined, but for performance
	// reasons we do not check for this here.
	NeuronID stackpos = spike_array_pos + (spike_attribute_offset+attribute_id)*src->get_spikes()->size();

	#ifdef DEBUG
	std::cout << "stack pos " << stackpos 
		<< " value: " << std::setprecision(5) 
		<< src->get_attributes()->at(stackpos) 
		<< std::endl;
	#endif //DEBUG

	return src->get_attributes()->at(stackpos);
}

DEFAULT_TRACE_MODEL * Connection::get_pre_trace(const AurynDouble tau)
{
	return src->get_pre_trace(tau);
}

DEFAULT_TRACE_MODEL * Connection::get_post_trace(const AurynDouble tau)
{
	return dst->get_post_trace(tau);
}

DEFAULT_TRACE_MODEL * Connection::get_post_state_trace(const string state_name, const AurynDouble tau, const AurynDouble jump_size)
{
	return dst->get_post_state_trace(state_name, tau, jump_size);
}


void Connection::evolve() 
{

}
