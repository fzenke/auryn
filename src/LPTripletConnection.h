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

#ifndef LPTRIPLETCONNECTION_H_
#define LPTRIPLETCONNECTION_H_

#include "auryn_definitions.h"
#include "DuplexConnection.h"
#include "EulerTrace.h"
#include "LinearTrace.h"
#include "SpikeDelay.h"

#define TRACE EulerTrace

using namespace std;


/*! \brief Implements triplet STDP in which weight updates are low-pass filtered.
 */
class LPTripletConnection : public DuplexConnection
{

private:
	friend class boost::serialization::access;
	template<class Archive>
	void save(Archive & ar, const unsigned int version) const
	{
		ar & boost::serialization::base_object<DuplexConnection>(*this);
	}
	template<class Archive>
	void load(Archive & ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<DuplexConnection>(*this);
		finalize();
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER()

	AurynFloat tau_lp;
	AurynFloat delta_lp;
	AurynTime timestep_lp;

	void init(AurynFloat tau_hom, AurynFloat eta, AurynFloat kappa, AurynFloat maxweight);
	void init_shortcuts();

	virtual AurynWeight get_hom(NeuronID i);

protected:

	AurynFloat tau_plus;
	AurynFloat tau_minus;
	AurynFloat tau_long;

	AurynFloat tau_homeostatic;

	NeuronID * fwd_ind; 
	AurynWeight * fwd_data;

	NeuronID * bkw_ind; 
	AurynWeight ** bkw_data;

	AurynDouble hom_fudge;

	/* Definitions of presynaptic traces */
	PRE_TRACE_MODEL * tr_pre;

	/* Definitions of postsynaptic traces */
	DEFAULT_TRACE_MODEL * tr_post;
	DEFAULT_TRACE_MODEL * tr_post2;
	DEFAULT_TRACE_MODEL * tr_post_hom;


	// temporary state vector
	AurynWeight * temp_state;

	void propagate_forward();
	void propagate_backward();

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


	/*! Toggle stdp active/inactive. When inactive traces are still updated, but weights are not. */
	bool stdp_active;

	LPTripletConnection(SpikingGroup * source, NeuronGroup * destination, 
			TransmitterType transmitter=GLUT);

	LPTripletConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat tau_hom=10, 
			AurynFloat eta=1, 
			AurynFloat kappa=3., AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT);

	/*! Default constructor. Sets up a random sparse connection and plasticity parameters
	 *
	 * @param source the presynaptic neurons.
	 * @param destinatino the postsynaptic neurons.
	 * @param weight the initial synaptic weight.
	 * @param sparseness the sparseness of the connection (probability of connection).
	 * @param tau_hom the timescale of the homeostatic rate estimate (moving average).
	 * @param eta the relaive learning rate (default=1).
	 * @param transmitter the TransmitterType (default is GLUT, glutamatergic).
	 * @param name a sensible identifier for the connection used in debug output.
	 */
	LPTripletConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05, 
			AurynFloat tau_hom=10, 
			AurynFloat eta=1, 
			AurynFloat kappa=3., AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT,
			string name = "LPTripletConnection" );

	virtual ~LPTripletConnection();
	virtual void finalize();
	void free();

	void set_hom_trace(AurynFloat freq);

	virtual void propagate();
	virtual void evolve();

};

#endif /*LPTRIPLETCONNECTION_H_*/
