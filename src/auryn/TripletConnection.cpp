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

#include "TripletConnection.h"

using namespace auryn;

void TripletConnection::init(AurynFloat tau_hom, AurynFloat eta, AurynFloat kappa, AurynFloat maxweight)
{
	if ( dst->get_post_size() == 0 ) return; // avoids to run this code on silent nodes with zero post neurons.

	/* Initialization of plasticity parameters. */
	A3_plus = 6.5e-3;
	A3_plus *= eta;

	tau_plus  = 16.8e-3;
	tau_minus = 33.7e-3;
	tau_long  = 114e-3;

	tau_homeostatic = tau_hom;

	/* Initialization of presynaptic traces */
	tr_pre = src->get_pre_trace(tau_plus);

	/* Initialization of postsynaptic traces */
	tr_post = dst->get_post_trace(tau_minus);
	tr_post2 = dst->get_post_trace(tau_long);
	tr_post_hom = dst->get_post_trace(tau_hom);

	hom_fudge = A3_plus*tau_plus*tau_long/(tau_minus)/kappa/tau_hom/tau_hom;

	/* Set min/max weight values. */
	set_min_weight(0.0);
	set_max_weight(maxweight);

	stdp_active = true;

}

void TripletConnection::init_shortcuts() 
{
	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

	fwd_ind = w->get_row_begin(0); 
	fwd_data = w->get_data_begin();

	bkw_ind = bkw->get_row_begin(0); 
	bkw_data = bkw->get_data_begin();
}

void TripletConnection::finalize() {
	DuplexConnection::finalize();
	init_shortcuts();
}

void TripletConnection::free()
{
}

TripletConnection::TripletConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) : DuplexConnection(source, destination, transmitter)
{
}

TripletConnection::TripletConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynFloat tau_hom, 
		AurynFloat eta, 
		AurynFloat kappa, AurynFloat maxweight , 
		TransmitterType transmitter) 
: DuplexConnection(source, 
		destination, 
		filename, 
		transmitter)
{
	init(tau_hom, eta, kappa, maxweight);
	init_shortcuts();
}

TripletConnection::TripletConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat tau_hom, 
		AurynFloat eta, 
		AurynFloat kappa, AurynFloat maxweight , 
		TransmitterType transmitter,
		std::string name) 
: DuplexConnection(source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name)
{
	init(tau_hom, eta, kappa, maxweight);
	if ( name.empty() ) 
		set_name("TripletConnection");
	init_shortcuts();
}

TripletConnection::~TripletConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}

void TripletConnection::set_hom_trace(AurynFloat freq)
{
	if ( dst->get_post_size() > 0 ) 
		tr_post_hom->set_all(freq*tr_post_hom->get_tau());
}


inline AurynWeight TripletConnection::get_hom(NeuronID i)
{
	return pow(tr_post_hom->get(i),2);
}


inline AurynWeight TripletConnection::dw_pre(NeuronID post)
{
	// translate post id to local id on rank: translated_spike
	NeuronID translated_spike = dst->global2rank(post); 
	AurynDouble dw = -hom_fudge*(tr_post->get(translated_spike)*get_hom(translated_spike));
	return dw;
}

inline AurynWeight TripletConnection::dw_post(NeuronID pre, NeuronID post)
{
	// at this point post was already translated to a local id in 
	// the propagate_backward function below.
	AurynDouble dw = A3_plus*tr_pre->get(pre)*tr_post2->get(post);
	return dw;
}


inline void TripletConnection::clip_weight( AurynWeight * weight ) 
{
	if (*weight<get_min_weight()) *weight = get_min_weight();
	else if (*weight>get_max_weight()) *weight=get_max_weight();
}


void TripletConnection::propagate_forward()
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

			// handles plasticity
			if ( stdp_active ) {
			    *weight += dw_pre(*c);
				clip_weight(weight);
			}

			// evokes the postsynaptic response 
			transmit( *c , *weight );
		}
	}
}

void TripletConnection::propagate_backward()
{
	if (stdp_active) { 
		// loop over all post spikes
		for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
				spike != dst->get_spikes_immediate()->end() ; 
				++spike ) {
			// translate the global post id to the neuron index on this rank
			NeuronID translated_spike = dst->global2rank(*spike); 

			// loop over all presynaptic partners
			for (const NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {

				// prefetches next memory cells to reduce number of last-level cache misses
				#ifdef CODE_ACTIVATE_PREFETCHING_INTRINSICS
				_mm_prefetch((const char *)bkw_data[c-bkw_ind+2],  _MM_HINT_NTA);
				#endif

				// computes plasticity update
				AurynWeight * weight = bkw->get_data(c); // for bkw data is already a pointer
				*weight += dw_post(*c,translated_spike);
			}
		}
	}
}

void TripletConnection::propagate()
{
	propagate_forward();
	propagate_backward();
}

void TripletConnection::evolve()
{
}


