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

#include "STDPConnection.h"

void STDPConnection::init(AurynFloat eta, AurynFloat maxweight)
{
	if ( dst->get_post_size() == 0 ) return;

	tau_pre1  = 20.0e-3;
	tau_pre2  = 200.0e-3;
	tau_post1 = 15.0e-3;
	tau_post2 = 200.0e-3;

	Apre  = eta; // pre-post
	Apost = eta; // post-pre

	Bpre  = -Apre*(tau_pre1/tau_pre2);
	Bpost = -Apost*(tau_post1/tau_post2);

	logger->parameter("eta",eta);
	logger->parameter("Apre",Apre);
	logger->parameter("Apost",Apost);
	logger->parameter("Bpre",Bpre);
	logger->parameter("Bpost",Bpost);

	tr_pre1  = src->get_pre_trace(tau_pre1);
	tr_pre2  = src->get_pre_trace(tau_pre2);
	tr_post1 = dst->get_post_trace(tau_post1);
	tr_post2 = dst->get_post_trace(tau_post2);

	w_min = 0.0;
	w_max = maxweight;

	stdp_active = true;

}

void STDPConnection::init_shortcuts() 
{
	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

	fwd_ind = w->get_row_begin(0); 
	fwd_data = w->get_data_begin();

	bkw_ind = bkw->get_row_begin(0); 
	bkw_data = bkw->get_data_begin();
}

void STDPConnection::finalize() {
	DuplexConnection::finalize();
	init_shortcuts();
}

void STDPConnection::free()
{
}

STDPConnection::STDPConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) : DuplexConnection(source, destination, transmitter)
{
}

STDPConnection::STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynFloat eta, 
		AurynFloat maxweight , 
		TransmitterType transmitter) 
: DuplexConnection(source, 
		destination, 
		filename, 
		transmitter)
{
	init(eta, maxweight);
	init_shortcuts();
}

STDPConnection::STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat eta, 
		AurynFloat maxweight , 
		TransmitterType transmitter,
		string name) 
: DuplexConnection(source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name)
{
	init(eta, maxweight);
	if ( name.empty() )
		set_name("STDPConnection");
	init_shortcuts();
}

STDPConnection::~STDPConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}


AurynWeight STDPConnection::dw_pre(NeuronID post)
{
	NeuronID translated_spike = dst->global2rank(post); // only to be used for post traces
	AurynDouble dw = Apost*tr_post1->get(translated_spike) + Bpost*tr_post2->get(translated_spike);
	return dw;
}

AurynWeight STDPConnection::dw_post(NeuronID pre, NeuronID post)
{
	AurynDouble dw = Apre*tr_pre1->get(pre) + Bpre*tr_pre2->get(pre);
	return dw;
}


void STDPConnection::propagate_forward()
{
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) { // c = post index
			AurynWeight value = fwd_data[c-fwd_ind]; 
			transmit( *c , value );
			if ( stdp_active ) {
			  fwd_data[c-fwd_ind] += dw_pre(*c);
			  if ( fwd_data[c-fwd_ind] < w_min ) 
				fwd_data[c-fwd_ind] = w_min;
			}
		}
		// update pre_trace
		// tr_pre->inc(*spike);
	}
}

void STDPConnection::propagate_backward()
{
	SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
	// process spikes
	for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
			spike != spikes_end ; ++spike ) {
		NeuronID translated_spike = dst->global2rank(*spike); // only to be used for post traces
		if (stdp_active) {
			for (NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {
			    _mm_prefetch(bkw_data[c-bkw_ind+1],  _MM_HINT_NTA); // tested this and this and NTA directly here gave the best performance on the SUN
				*bkw_data[c-bkw_ind] = *bkw_data[c-bkw_ind] + dw_post(*c,translated_spike);
				if (*bkw_data[c-bkw_ind]>w_max) *bkw_data[c-bkw_ind]=w_max;
			}
		}
	}
}

void STDPConnection::propagate()
{
	propagate_forward();
	propagate_backward();
}

void STDPConnection::evolve()
{
}

void STDPConnection::set_min_weight(AurynWeight min)
{	
	w_min = min;
}

void STDPConnection::set_max_weight(AurynWeight max)
{	
	w_max = max;
}

AurynWeight STDPConnection::get_wmin()
{
	return w_min;
}

