/* 
* Copyright 2014 Friedemann Zenke
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
*/

#include "TripletConnection.h"

void TripletConnection::init(AurynFloat tau_hom, AurynFloat eta, AurynFloat kappa, AurynFloat maxweight)
{
	if ( dst->get_post_size() == 0 ) return;

	A3_plus = 6.5e-3;
	A3_plus *= eta;

	tau_plus  = 16.8e-3;
	tau_minus = 33.7e-3;
	tau_long  = 114e-3;

	tau_homeostatic = tau_hom;

	tr_pre = src->get_pre_trace(tau_plus);
	// tr_pre = new LinearTrace(src->get_pre_size(),tau_plus); 
	tr_post = dst->get_post_trace(tau_minus);
	tr_post2 = dst->get_post_trace(tau_long);
	tr_post_hom = dst->get_post_trace(tau_hom);

	hom_fudge = A3_plus*tau_plus*tau_long/(tau_minus)/kappa/tau_hom/tau_hom;
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
		string name) 
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


AurynWeight TripletConnection::get_hom(NeuronID i)
{
	return pow(tr_post_hom->get(i),2);
}


AurynWeight TripletConnection::dw_pre(NeuronID post)
{
	NeuronID translated_spike = dst->global2rank(post); // only to be used for post traces
	AurynDouble dw = hom_fudge*(tr_post->get(translated_spike)*get_hom(translated_spike));
	// cout << "pre" << dw << endl;
	return dw;
}

AurynWeight TripletConnection::dw_post(NeuronID pre, NeuronID post)
{
	// post translation is done in loop below
	AurynDouble dw = A3_plus*tr_pre->get(pre)*tr_post2->get(post);
	// cout << "post" << dw << endl;
	return dw;
}


void TripletConnection::propagate_forward()
{
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		for (const NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) { // c = post index
			AurynWeight value = fwd_data[c-fwd_ind]; 
			transmit( *c , value );
			if ( stdp_active ) {
			  fwd_data[c-fwd_ind] -= dw_pre(*c);
			  if ( fwd_data[c-fwd_ind] < get_min_weight() ) 
				fwd_data[c-fwd_ind] = get_min_weight();
			}
		}
		// update pre_trace
		// tr_pre->inc(*spike);
	}
}

void TripletConnection::propagate_backward()
{
	if (stdp_active) {
		SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
		// process spikes
		for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
				spike != spikes_end ; ++spike ) {
			NeuronID translated_spike = dst->global2rank(*spike); // only to be used for post traces
			for (const NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {
				_mm_prefetch(bkw_data[c-bkw_ind+2],  _MM_HINT_NTA); 
				*bkw_data[c-bkw_ind] = *bkw_data[c-bkw_ind] + dw_post(*c,translated_spike);
				if (*bkw_data[c-bkw_ind]>get_max_weight()) *bkw_data[c-bkw_ind]=get_max_weight();
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


