/* 
* Copyright 2014-2018 Friedemann Zenke
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

#ifndef BCPNNCONNECTION_H_
#define BCPNNCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "DuplexConnection.h"
#include "Trace.h"
#include "LinearTrace.h"
#include "SpikeDelay.h"


namespace auryn {

/*! \brief Implements triplet STDP with metaplasticity as described by Pfister and Gerstner 2006.
 *
 * This is the connection used to simulate large plastic recurrent networks with the BCPNN learning rule.
 *
 */
class BcpnnConnection : public DuplexConnection
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

	void init(AurynFloat tau_pre,AurynFloat tau_z_pre, AurynFloat tau_z_post, AurynFloat tau_p,
			  AurynFloat refractory_period) ;
	void init_shortcuts();

protected:

	AurynFloat tau_z_pre;
	AurynFloat tau_z_post;
	AurynFloat tau_p;

	AurynFloat kinc_z_pre; 
	AurynFloat kinc_z_post; 
	AurynFloat kinc_p; 

	AurynFloat p_decay;

	int zid_wij,zid_pi,zid_pij;

	AurynState *bj_vectordata;
	AurynVector<float,long unsigned int> *wij_vector;
	AurynVector<float,long unsigned int> *pij_vector;
	AurynVector<float,long unsigned int> *pi_vector;

	NeuronID * fwd_ind; 
	AurynWeight * fwd_data;

	NeuronID * bkw_ind; 
	AurynWeight ** bkw_data;

	/*! \brief Propagates spikes from pre to post
	 *
	 *
	 * This function propagates spikes from pre to postsynaptic cells
	 * and performs plasticity updates upon presynaptic spikes. */
	void propagate_forward();

public:

	/* Definitions of presynaptic traces */
	Trace * tr_pre;
	Trace * tr_z_pre;

	/* Definitions of postsynaptic traces */
	Trace * tr_post;
	Trace * tr_z_post;
	Trace * tr_p_post;

	AurynFloat eps;
	AurynFloat eps2;
	AurynFloat bgain;
	AurynFloat wgain;

	/*! \brief Toggles stdp active/inactive. When inactive traces are still updated, but weights are not. */
	bool stdp_active;

	/*! \brief Empty connection constructor.
	 *
	 * Does not initialize connection with random sparse connectivity.
	 * \see SparseConnection::connect_random 
	 */
	BcpnnConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter=GLUT);

	/*! \brief Default constructor. Sets up a random sparse connection and plasticity parameters
	 *
	 * \param source the presynaptic neurons.
	 * \param destinatino the postsynaptic neurons.
	 * \param weight the initial synaptic weight.
	 * \param sparseness the sparseness of the connection (probability of connection).
	 * \param time constant for epsp
	 * \param time constant for z_pre_trace
	 * \param time constant for z_post_trace
	 * \param time constant for p_trace
	 * \param transmitter the TransmitterType (default is GLUT, glutamatergic).
	 * \param name a sensible identifier for the connection used in debug output.
	 */
    BcpnnConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat tau_pre = 0.010, 
		AurynFloat tau_z_pre = 0.010, 
		AurynFloat tau_z_post = 0.010,
		AurynFloat tau_p = 1.0, 
		AurynFloat refractory_period = 0.005,
		TransmitterType transmitter = GLUT,
		std::string name = "BcpnnConnection");

	
	virtual ~BcpnnConnection();
	virtual void finalize();
	void free();

	virtual void propagate();
	virtual void evolve();

	void set_eps(AurynFloat value) ;
	void set_bgain(AurynFloat value) ;
	void set_wgain(AurynFloat value) ;

};

}

#endif /*BCPNNCONNECTION_H_*/
