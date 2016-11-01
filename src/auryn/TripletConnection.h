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

#ifndef TRIPLETCONNECTION_H_
#define TRIPLETCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "DuplexConnection.h"
#include "Trace.h"
#include "LinearTrace.h"
#include "SpikeDelay.h"


namespace auryn {

/*! \brief Implements triplet STDP with metaplasticity as described by Pfister and Gerstner 2006.
 *
 * This is the connection used for most simulations in Zenke et al. 2013 to 
 * simulate large plastic recurrent networks with homeostatic triplet STDP.
 * Time timescale of the moving average used for the homeostatic change of the
 * LTD rate is given by tau.
 *
 * \see sim_background.cpp
 */
class TripletConnection : public DuplexConnection
{

private:
	void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
	{
		DuplexConnection::virtual_serialize(ar,version);
	}

	void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
	{
		DuplexConnection::virtual_serialize(ar,version);
		DuplexConnection::compute_reverse_matrix(); // just in case the buffer location has changed 
	}

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

	/*! \brief Clips weight to allowed range
	 *
	 * \param weight a pointer to the weight value to be clipped 
	 *
	 * This clips a weight referenced by pointer to the ranke get_min_weight() < weight < get_max_weight().
	 * \see get_min_weight, get_max_weight
	 *
	 *
	 * */
	void clip_weight( AurynWeight * weight );

	/*! \brief Propagates spikes from pre to post
	 *
	 *
	 * This function propagates spikes from pre to postsynaptic cells
	 * and performs plasticity updates upon presynaptic spikes. */
	void propagate_forward();

	/*! \brief Back-propagates spikes from post to pre 
	 *
	 * This performs plasticity updates following postsynaptic spikes. To that end the postsynaptic spikes 
	 * have to be communicated backward to the corresponding synapses connecting to presynaptic neurons. This
	 * is why this function is called propagate_backward ... it is remeniscent of a back-propagating action 
	 * potential. */
	void propagate_backward();


	/*! \brief Basic event-based weight update upon presynaptic spike
	 *
	 *
	 *  \param post the parameter specifies the postsynaptic partner for which we 
	 *  are computing the update. 
	 *  \returns weight update
	 *
	 * This function implements the plastic update to each 
	 *  synapse at the time of a presynaptic spike.
	 *
	 *  */
	AurynWeight dw_pre(NeuronID post);

	/*! \brief Event-based weight update upon postsynaptic spike time.
	 *
	 *  \param pre The parameter specifies the presynaptic partner for which we 
	 *  are computing the update. 
	 *  \param post the parameter specifies the postsynaptic partner for which we 
	 *  are computing the update. 
	 *  \returns weight update
	 *
	 * This function implements the plastic update to each 
	 *  synapse at the time of a postsynaptic spike. Since LTP in the minimal triplet model
	 *  depends on the timing of the last pre and postsynaptic spike we are passing both NeuronID 
	 *  as arguments.
	 *
	 *  */
	AurynWeight dw_post(NeuronID pre, NeuronID post);


public:
	AurynFloat A3_plus;

	/* Definitions of presynaptic traces */
	Trace * tr_pre;

	/* Definitions of postsynaptic traces */
	Trace * tr_post;
	Trace * tr_post2;
	Trace * tr_post_hom;

	/*! \brief Toggles stdp active/inactive. When inactive traces are still updated, but weights are not. */
	bool stdp_active;

	/*! \brief Empty connection constructor.
	 *
	 * Does not initialize connection with random sparse connectivity.
	 * \see SparseConnection::connect_random 
	 */
	TripletConnection(SpikingGroup * source, NeuronGroup * destination, 
			TransmitterType transmitter=GLUT);

	/*! \brief \deprecated Deprecated constructor.
	 */
	TripletConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat tau_hom=10, 
			AurynFloat eta=1, 
			AurynFloat kappa=3., AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT);

	/*! \brief Default constructor. Sets up a random sparse connection and plasticity parameters
	 *
	 * \param source the presynaptic neurons.
	 * \param destinatino the postsynaptic neurons.
	 * \param weight the initial synaptic weight.
	 * \param sparseness the sparseness of the connection (probability of connection).
	 * \param tau_hom the timescale of the homeostatic rate estimate (moving average).
	 * \param eta the relaive learning rate (default=1).
	 * \param transmitter the TransmitterType (default is GLUT, glutamatergic).
	 * \param name a sensible identifier for the connection used in debug output.
	 */
	TripletConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05, 
			AurynFloat tau_hom=10, 
			AurynFloat eta=1, 
			AurynFloat kappa=3., AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT,
			string name = "TripletConnection" );

	virtual ~TripletConnection();
	virtual void finalize();
	void free();

	void set_hom_trace(AurynFloat freq);

	virtual void propagate();
	virtual void evolve();

};

}

#endif /*TRIPLETCONNECTION_H_*/
