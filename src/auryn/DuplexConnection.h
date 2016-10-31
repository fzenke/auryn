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

#ifndef DUPLEXCONNECTION_H_
#define DUPLEXCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "SparseConnection.h"
#include "SimpleMatrix.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

namespace auryn {

/*! Definition of BackwardMatrix - a sparsematrix of pointers to weight values. */
typedef SimpleMatrix<AurynWeight*> BackwardMatrix;

/*! \brief Duplex connection is the base class of most plastic connections.
 * 
 * DuplexConnection serves as base class for plastic connections that want to implement
 * synaptic weight change in response to a postsynaptic spike. This requires the post
 * spike to be signalled back to the presynaptic cell (rather the connecting synapse).
 * To do this efficiently, the weight matrix (ForwardMatrix) which allows for efficient 
 * forward propagation of spikes is mirrored as its transposed (BackwardMatrix). 
 * To keep the two matrices in sync the BackwardMatrix does not contain the actual weight
 * value to the forward weight, but a pointer reference to that value.
 */
class DuplexConnection : public SparseConnection
{
private:
	bool allocated_bkw;
	void init();
	void free();
protected:
	void compute_reverse_matrix(int z = 0);
public:
	ForwardMatrix  * fwd;
	BackwardMatrix * bkw; // TODO make protected again later when tested

	DuplexConnection(const char * filename);
	DuplexConnection(NeuronID rows, NeuronID cols);
	DuplexConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter=GLUT);
	DuplexConnection(SpikingGroup * source, NeuronGroup * destination, const char * filename , TransmitterType transmitter=GLUT);
	DuplexConnection(SpikingGroup * source, NeuronGroup * destination, AurynWeight weight, AurynFloat sparseness=0.05, TransmitterType transmitter=GLUT, std::string name="DuplexConnection");


	virtual ~DuplexConnection();
	virtual void finalize();

	/*! \brief Prune weight matrices. */
	void prune();

};

}

#endif /*DUPLEXCONNECTION_H_*/
