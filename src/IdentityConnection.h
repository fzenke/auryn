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

#ifndef IDENTITYCONNECTION_H_
#define IDENTITYCONNECTION_H_

#include "auryn_definitions.h"
#include "Connection.h"
#include "System.h"

#include <sstream>
#include <fstream>
#include <stdio.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

namespace auryn {

/*! \brief Provides a unity matrix like connectivity */
class IdentityConnection : public Connection
{
private:
	SpikeContainer * spikes;
	void init(AurynWeight weight);
	AurynWeight connection_weight;
	NeuronID lo_src,hi_src,lo_dst,hi_dst;
	NeuronID offset;
	NeuronID every;

protected:
	void free();

	
public:
	IdentityConnection();
	IdentityConnection(SpikingGroup * source, NeuronGroup * destination, AurynWeight weight = 1.0, TransmitterType transmitter = GLUT, string name = "Default IdentityConnection");
	IdentityConnection(SpikingGroup * source, NeuronGroup * destination, AurynWeight weight, NeuronID lo_row, NeuronID hi_row, NeuronID lo_col, NeuronID hi_col, TransmitterType transmitter=GLUT, string name = "Cropped Identity Connection" );
	virtual ~IdentityConnection();

	virtual AurynWeight get(NeuronID i, NeuronID j);
	virtual AurynWeight * get_ptr(NeuronID i, NeuronID j);
	virtual AurynWeight get_data(NeuronID i);
	void set_offset(int off);
	void set_every(NeuronID e);
	virtual void set_data(NeuronID i, AurynWeight value);
	virtual void set(NeuronID i, NeuronID j, AurynWeight value);
	void finalize();
	AurynLong get_nonzero();
	virtual void propagate();

	virtual AurynDouble sum();
	virtual void stats(AurynDouble &mean, AurynDouble &std);
	virtual bool write_to_file(string filename);
	virtual bool load_from_file(string filename);
	virtual AurynFloat mean();

	/*! Returns a vector of ConnectionsID of a block specified by the arguments */
	std::vector<neuron_pair> get_block(NeuronID lo_row, NeuronID lo_col, NeuronID hi_row,  NeuronID hi_col);

};

}

#endif /*IDENTITYCONNECTION_H_*/
