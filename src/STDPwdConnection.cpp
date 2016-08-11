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

#include "STDPwdConnection.h"

using namespace auryn;

void STDPwdConnection::init(AurynWeight lambda, AurynWeight maxweight)
{
	if ( dst->get_post_size() == 0 ) return;

	tau_plus  = 20.0e-3;
	tau_minus = 20.0e-3;


	tr_pre  = src->get_pre_trace(tau_minus);
	tr_post = dst->get_post_trace(tau_plus);

	set_max_weight(maxweight); 

	param_lambda = lambda;
	param_alpha  = 1.0;
	param_mu_plus = 1.0;
	param_mu_minus = 1.0;


	compute_fudge_factors();

	stdp_active = true;
}

void STDPwdConnection::compute_fudge_factors()
{
	learning_rate = param_lambda;
	fudge_dep = learning_rate*pow(get_max_weight(),1.0-param_mu_minus)*param_alpha;
	fudge_pot = learning_rate*pow(get_max_weight(),1.0-param_mu_plus);
}

void STDPwdConnection::init_shortcuts() 
{
	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

	fwd_ind = w->get_row_begin(0); 
	fwd_data = w->get_data_begin();

	bkw_ind = bkw->get_row_begin(0); 
	bkw_data = bkw->get_data_begin();
}

void STDPwdConnection::finalize() {
	DuplexConnection::finalize();
	init_shortcuts();
}

void STDPwdConnection::free()
{
}

STDPwdConnection::STDPwdConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) : DuplexConnection(source, destination, transmitter)
{
}

STDPwdConnection::STDPwdConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynWeight lambda, 
		AurynWeight maxweight , 
		TransmitterType transmitter) 
: DuplexConnection(source, 
		destination, 
		filename, 
		transmitter)
{
	init(lambda, maxweight);
	init_shortcuts();
}

STDPwdConnection::STDPwdConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynWeight sparseness, 
		AurynWeight lambda, 
		AurynWeight maxweight , 
		TransmitterType transmitter,
		std::string name) 
: DuplexConnection(source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name)
{
	init(lambda, maxweight);
	if ( name.empty() )
		set_name("STDPwdConnection");
	init_shortcuts();
}

STDPwdConnection::~STDPwdConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}


void STDPwdConnection::propagate_forward()
{
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) { // c = post index
			// create a shortcut for readability
		    AurynWeight * value = &fwd_data[c-fwd_ind]; 

			// clip weight
			if ( *value < get_min_weight() ) *value = get_min_weight() ;
			else if ( *value > get_max_weight() ) *value = get_max_weight() ;

			transmit( *c , *value );

			// STDP update post-pre
			if ( stdp_active ) {
			  NeuronID translated_spike = dst->global2rank(*c); // get ID of neuron on rank
			  AurynWeight fminus = pow(*value,param_mu_minus); // compute f_minus(w) function value
			  *value += -fudge_dep*fminus*tr_post->get(translated_spike); // update the weight
			}
		}
		// update pre_trace
		// tr_pre->inc(*spike);
	}
}

void STDPwdConnection::propagate_backward()
{
	SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
	// process spikes
	for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
			spike != spikes_end ; ++spike ) {
		if (stdp_active) {
			for (NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {

				#ifdef CODE_ACTIVATE_PREFETCHING_INTRINSICS
				_mm_prefetch((const char *)bkw_data[c-bkw_ind+1],  _MM_HINT_NTA);
				#endif

				AurynWeight * value = bkw_data[c-bkw_ind]; // create a shortcut for readability
			    AurynWeight fplus = pow((get_max_weight()-*value),param_mu_plus); // compute f_minus(w) function value
			    *value += fudge_pot*fplus*tr_pre->get(*c); // update the weight
			}
		}
	}
}

void STDPwdConnection::set_alpha(AurynWeight a)
{
	param_alpha = a;
	compute_fudge_factors();
	auryn::logger->parameter("alpha",param_alpha);
}

void STDPwdConnection::set_lambda(AurynWeight l)
{
	param_lambda = l;
	compute_fudge_factors();
	auryn::logger->parameter("lambda",param_lambda);
}

void STDPwdConnection::set_mu_plus(AurynWeight m)
{
	param_mu_plus = m;
	compute_fudge_factors();
	auryn::logger->parameter("mu_plus",param_mu_plus);
}

void STDPwdConnection::set_mu_minus(AurynWeight m)
{
	param_mu_minus = m;
	compute_fudge_factors();
	auryn::logger->parameter("mu_minus",param_mu_minus);
}

void STDPwdConnection::set_max_weight(AurynWeight wmax)
{
	SparseConnection::set_max_weight(wmax);
	compute_fudge_factors();
}

void STDPwdConnection::propagate()
{
	propagate_forward();
	propagate_backward();
}

void STDPwdConnection::evolve()
{
	//tr_pre->evolve();
}

