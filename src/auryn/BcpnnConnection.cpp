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

#include "BcpnnConnection.h"

using namespace auryn;

void BcpnnConnection::init(AurynFloat tau_pre,AurynFloat tau_z_pre, AurynFloat tau_z_post, AurynFloat tau_p,AurynFloat refractory_period)
//void BcpnnConnection::init(AurynFloat tau_hom, AurynFloat eta, AurynFloat kappa, AurynFloat maxweight)
{
	if ( dst->get_post_size() == 0 ) return; // avoids to run this code on silent nodes with zero post neurons.

	/* Initialization of plasticity parameters. */

	this->tau_z_pre = tau_z_pre;
	this->tau_z_post = tau_z_post;
	this->tau_p = tau_p;

	/* Initialization of presynaptic traces */
	tr_pre = src->get_pre_trace(tau_pre);

	// bcpnn_zi trace

	kinc_z_pre = refractory_period/tau_z_pre;
	tr_z_pre = src->get_pre_trace(tau_z_pre);
	tr_z_pre->set_kinc(kinc_z_pre);

	// bcpnn_pi trace
    kinc_p = 1 - exp(-auryn_timestep/tau_p);
	tr_p_pre = src->get_post_state_trace(tr_z_pre,tau_p);
	tr_p_pre->set_kinc(kinc_p);
	src->add_state_vector("tr_p_pre",tr_p_pre);

	/* Initialization of postsynaptic traces */

	// bcpnn_zj trace
	kinc_z_post = refractory_period/tau_z_post;
	tr_z_post = dst->get_post_trace(tau_z_post); 
	tr_z_post->set_kinc(kinc_z_post);

	// bcpnn_pj trace
	tr_p_post = dst->get_post_state_trace(tr_z_post,tau_p);
	tr_p_pre->set_kinc(kinc_p);
	dst->add_state_vector("tr_p_post",tr_p_post);
	
	// bcpnn_pij trace

	w->set_num_synapse_states(2);
	zid_wij = 0;
	zid_pij= 1;

	p_decay = exp(-auryn_timestep/tau_p);
 
	set_eps(1e-12);
	set_bgain(1);
	set_wgain(1);

	bias_variable= dst->get_state_vector("w")->data;

	stdp_active = true;

}

void BcpnnConnection::set_eps(AurynFloat value) {

	try {
		if (value<sqrt(std::numeric_limits<float>::min()))
			throw std::invalid_argument("");
	} catch (std::invalid_argument iaex) {
		std::stringstream oes;
		oes << "Too small eps = " << value ;
		auryn::logger->msg(oes.str(),ERROR);
	   auryn_abort(11);
	}
	eps = value;
	eps2 = eps * eps;

}

void BcpnnConnection::set_bgain(AurynFloat value) {

	bgain = value;

}
void BcpnnConnection::set_wgain(AurynFloat value) {

	wgain = value;

}


void BcpnnConnection::init_shortcuts() 
{
	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

	fwd_ind = w->get_row_begin(0); 
	fwd_data = w->get_data_begin();

	bkw_ind = bkw->get_row_begin(0); 
	bkw_data = bkw->get_data_begin();
}

void BcpnnConnection::finalize() {
	DuplexConnection::finalize();
	init_shortcuts();
}

void BcpnnConnection::free()
{
	std::stringstream oss;
	oss << "BcpnnConnection:: " << 
		get_log_name()
		<< " freeing ...";
	auryn::logger->msg(oss.str(),VERBOSE);
	auryn::logger->msg("BcpnnConnection:: Freeing poststatetraces",VERBOSE);
	src->remove_state_vector("tr_p_pre");
	dst->remove_state_vector("tr_p_post");

}

BcpnnConnection::BcpnnConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) :
	DuplexConnection(source, destination, transmitter)
{
}

BcpnnConnection::BcpnnConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat tau_pre, 
		AurynFloat tau_z_pre, 
		AurynFloat tau_z_post,
		AurynFloat tau_p, 
		AurynFloat refractory_period,
		TransmitterType transmitter,
		std::string name) 
: DuplexConnection(source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name)
{
	init(tau_pre,tau_z_pre,tau_z_post,tau_p,refractory_period);
	if ( name.empty() ) 
		set_name("BcpnnConnection");
	init_shortcuts();
}

BcpnnConnection::~BcpnnConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}

void BcpnnConnection::propagate_forward()
{
	// loop over all pre spikes 
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		// loop over all postsynaptic target cells 
		for (const NeuronID * c = w->get_row_begin(*spike) ; 
				c != w->get_row_end(*spike) ; 
				++c ) { // c = post index

			// determines the weight of connection
			AurynWeight * weight = w->get_data_ptr(c); 

			// evokes the postsynaptic response 
			transmit(*c , *weight);
		}
	}
}

#define USED

void BcpnnConnection::propagate()
{
#ifdef USED
	propagate_forward();
#endif // USED
}

void BcpnnConnection::evolve() {

#ifdef USED

	w->get_state_vector(zid_pij)->scale(p_decay);

	// add pre*post elements
	for (NeuronID li = 0; li < dst->get_post_size() ; ++li ) {
		const NeuronID gi = dst->rank2global(li); // id translation for MPI
		const AurynWeight pj = tr_p_post->get(li);

		/* Updating postneuron bias bj */
		bias_variable[li] = bgain * std::log(pj + eps);

		for (const NeuronID *c = bkw->get_row_begin(gi) ; 
			 c != bkw->get_row_end(gi) ; 
			 ++c ) {
			AurynWeight *weight = bkw->get_data(c); 
			const AurynLong didx = w->data_ptr_to_didx(weight); // index of element in data array
			const AurynState zi = tr_z_pre->get(*c);
			const AurynState zj = tr_z_post->get(li);
			AurynState de = kinc_p * zi * zj; // our multiplication
			w->get_state_vector(zid_pij)->add_specific(didx,de);

			const AurynWeight pi = tr_p_pre->get(*c);
			const AurynWeight pij = w->get_state_vector(zid_pij)->get(didx);
			const AurynWeight wij = wgain * std::log((pij + eps2)/((pi + eps) * (pj + eps)));
			w->get_state_vector(zid_wij)->set(didx,wij);
		}

	}

#endif // USED

}

