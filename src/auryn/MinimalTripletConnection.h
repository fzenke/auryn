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

#ifndef MINIMALTRIPLETCONNECTION_H_
#define MINIMALTRIPLETCONNECTION_H_

#include "auryn_definitions.h"
#include "DuplexConnection.h"
#include "EulerTrace.h"
#include "LinearTrace.h"
#include "SpikeDelay.h"

#define TRACE EulerTrace


namespace auryn {

/*! \brief Implements minimal triplet STDP as described by Pfister and Gerstner 2006.
 *
 */
class MinimalTripletConnection : public DuplexConnection
{

private:
	void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
	{
		DuplexConnection::virtual_serialize(ar,version);
		ar & *w;
	}

	void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
	{
		DuplexConnection::virtual_serialize(ar,version);
		ar & *w;
	}

	void init(AurynFloat eta, AurynFloat maxweight);
	void init_shortcuts();

protected:

	AurynFloat tau_plus;
	AurynFloat tau_minus;
	AurynFloat tau_long;


	NeuronID * fwd_ind; 
	AurynWeight * fwd_data;

	NeuronID * bkw_ind; 
	AurynWeight ** bkw_data;


	/* Definitions of presynaptic traces */
	PRE_TRACE_MODEL * tr_pre;

	/* Definitions of postsynaptic traces */
	DEFAULT_TRACE_MODEL * tr_post;
	DEFAULT_TRACE_MODEL * tr_post2;

	void propagate_forward();
	void propagate_backward();
	void sort_spikes();

	/*! Action on weight upon presynaptic spike on connection with postsynaptic
	 * partner post. This function should be modified to define new spike based
	 * plasticity rules. 
	 * @param post the postsynaptic cell from which the synaptic trace is read out*/
	AurynWeight dw_pre(NeuronID post);

	/*! Action on weight upon postsynaptic spike of cell post on connection
	 * with presynaptic partner pre. This function should be modified to define
	 * new spike based plasticity rules. 
	 * @param pre the presynaptic cell in question.
	 * @param post the postsynaptic cell in question. 
	 */ 
	AurynWeight dw_post(NeuronID pre, NeuronID post);


public:
	AurynFloat A3_plus;
	AurynFloat A2_minus;
	AurynFloat A2_plus;


	/*! Toggle stdp active/inactive. When inactive traces are still updated, but weights are not. */
	bool stdp_active;

	MinimalTripletConnection(SpikingGroup * source, NeuronGroup * destination, 
			TransmitterType transmitter=GLUT);

	/*! Deprecated constructor.
	 */
	MinimalTripletConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			const char * filename, 
			AurynFloat eta=1, 
			AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT);

	/*! Default constructor. Sets up a random sparse connection and plasticity parameters
	 *
	 * @param source the presynaptic neurons.
	 * @param destinatino the postsynaptic neurons.
	 * @param weight the initial synaptic weight.
	 * @param sparseness the sparseness of the connection (probability of connection).
	 * @param eta the relaive learning rate (default=1).
	 * @param transmitter the TransmitterType (default is GLUT, glutamatergic).
	 * @param name a sensible identifier for the connection used in debug output.
	 */
	MinimalTripletConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			AurynWeight weight, 
			AurynFloat sparseness=0.05, 
			AurynFloat eta=1.0, 
			AurynFloat maxweight=1.0, 
			TransmitterType transmitter=GLUT,
			string name = "MinimalTripletConnection" );

	virtual ~MinimalTripletConnection();
	virtual void finalize();
	void free();


	virtual void propagate();
	virtual void evolve();

};

}

#endif /*MINIMALTRIPLETCONNECTION_H_*/
