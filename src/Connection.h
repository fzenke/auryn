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

#ifndef CONNECTION_H_
#define CONNECTION_H_

#include <string>

#include "auryn_definitions.h"
#include "SpikingGroup.h"
#include "NeuronGroup.h"

class System;

/*! \brief The abstract base class for all Connection objects in Auryn
 *
 * Connections are designed to take up spikes from a source a group that
 * can spike in general (decendants of SpikingGroup) and to convey them
 * to a group that can integrate spikes such as the children of 
 * NeuronGroup.
 *
 * For that reason, when constructing any decendant of Connection you 
 * generally will have to specify a source (SpikingGroup) and a 
 * destination (NeuronGroup).
 */
class Connection 
{
private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		virtual_serialize(ar, version);
	}


	NeuronID m_rows,n_cols;
	string connection_name;

protected:
	/*! Serialization function for saving the Connection state. Implement in derived classes to save
	 * additional information. 
	 * */
	virtual void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
	{
		ar & m_rows & n_cols & connection_name;
	}

	/*! Serialization function for loading the Connection state. Implement in derived classes to save
	 * additional information. 
	 * */
	virtual void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
	{
		ar & m_rows & n_cols & connection_name;
	}

	SpikingGroup * src;
	NeuronGroup * dst;
	TransmitterType trans;
	AurynFloat * target;

public:
	Connection();
	Connection(NeuronID rows, NeuronID cols);
	Connection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter=GLUT, string name="Connection");
	virtual ~Connection();
	void init(TransmitterType transmitter=GLUT);

	void set_size(NeuronID i, NeuronID j);
	void set_name(string name);
	string get_name();

	NeuronID get_m_rows();
	NeuronID get_n_cols();
	TransmitterType get_transmitter();
	void set_transmitter(AurynWeight * ptr);
	void set_transmitter(TransmitterType transmitter);

	void set_source(SpikingGroup * source);
	SpikingGroup * get_source();
	void set_destination(NeuronGroup * source);
	NeuronGroup * get_destination();

	virtual AurynWeight get(NeuronID i, NeuronID j) = 0;
	virtual AurynWeight * get_ptr(NeuronID i, NeuronID j) = 0;
	virtual AurynWeight get_data(NeuronID i) = 0;
	virtual void set(NeuronID i, NeuronID j, AurynWeight value) = 0;
	virtual AurynLong get_nonzero() = 0;

	virtual void finalize() = 0;
	virtual void propagate() = 0;
	virtual void evolve();

	/*! DEPRECATED. (Such connections should not be registered in the first place) Calls propagate only if the postsynaptic NeuronGroup exists on the local rank. */
	void conditional_propagate();

	/*! Computes the sum of all weights in the connection. */
	virtual AurynDouble sum() = 0;

	/*! Computes mean synaptic weight and std dev of all weights in this connection. */
	virtual void stats(AurynFloat &mean, AurynFloat &std) = 0;

	/*! Implements save to file functionality. Also called in save_network_state from System class. */
	virtual bool write_to_file(string filename) = 0;

	/*! Implements load from file functionality. Also called in save_network_state from System class. */
	virtual bool load_from_file(string filename) = 0;

	/*! This is a new approach towards replacing tadd, increments transmitter specific state variables in neuron i*/
	inline void transmit(NeuronID id, AurynWeight amount);

	/*! Same as transmit but checks if the target neuron exists */
	void safe_transmit(NeuronID id, AurynWeight amount);

	/*! Returns a vector of ConnectionsID of a block specified by the arguments */
	virtual vector<neuron_pair>  get_block(NeuronID lo_row, NeuronID lo_col, NeuronID hi_row,  NeuronID hi_col) = 0;

};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Connection)

inline void Connection::transmit(NeuronID id, AurynWeight amount) 
{
	NeuronID localid = dst->global2rank(id);
	target[localid]+=amount;
}

#endif /*CONNECTION_H_*/
