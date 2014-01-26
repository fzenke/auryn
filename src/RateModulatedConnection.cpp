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

#include "RateModulatedConnection.h"

void RateModulatedConnection::init() 
{
	rate_target = 5;
	rate_estimate = rate_target;
	rate_estimate_tau = 1.0;
	rate_estimate_decay_mul = exp(-dt/rate_estimate_tau);
	rate_modulation_exponent = 2;
	rate_modulation_mul = 0.0;
	rate_modulating_group = NULL;

	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank
	set_modulating_group(dst);

	// weight maxima
	w_min = 0.0;
	w_max = 5.0;

	// Synaptic traces
	eta = 1e-3; // learning rate
	tau_stdp = 20.0e-3; // STDP window size (symmetric)
	tr_pre = src->get_pre_trace(tau_stdp);
	tr_post = dst->get_post_trace(tau_stdp);

	init_shortcuts();
}

void RateModulatedConnection::init_shortcuts() 
{
	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

	fwd_ind = w->get_row_begin(0); 
	fwd_data = w->get_data_begin();

	bkw_ind = bkw->get_row_begin(0); 
	bkw_data = bkw->get_data_begin();
}

RateModulatedConnection::RateModulatedConnection(const char * filename) 
: DuplexConnection(filename)
{
	init();
}

RateModulatedConnection::RateModulatedConnection(SpikingGroup * source, NeuronGroup * destination, 
		TransmitterType transmitter) 
: DuplexConnection(source, destination, transmitter)
{
}

RateModulatedConnection::RateModulatedConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename , 
		TransmitterType transmitter) 
: DuplexConnection(source, destination, filename, transmitter)
{
	init();
}


RateModulatedConnection::RateModulatedConnection(NeuronID rows, NeuronID cols) 
: DuplexConnection(rows,cols)
{
	init();
}

RateModulatedConnection::RateModulatedConnection( SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		TransmitterType transmitter, string name) 
: DuplexConnection(source,destination,weight,sparseness,transmitter, name)
{
	init();
}

void RateModulatedConnection::free()
{
}

AurynWeight RateModulatedConnection::dw_pre(NeuronID post)
{
	NeuronID translated_spike = dst->global2rank(post); 
	AurynDouble dw = rate_modulation_mul*tr_post->get(translated_spike)-1e-3*eta;
	return dw;
}

AurynWeight RateModulatedConnection::dw_post(NeuronID pre)
{
	AurynDouble dw = rate_modulation_mul*tr_pre->get(pre);
	return dw;
}

void RateModulatedConnection::propagate_forward()
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
	}
}

void RateModulatedConnection::propagate_backward()
{
	SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
	// process spikes
	for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
			spike != spikes_end ; ++spike ) {
		if (stdp_active) {
			for (NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {
			    _mm_prefetch(bkw_data[c-bkw_ind+1],  _MM_HINT_NTA); // tested this and this and NTA directly here gave the best performance on the SUN
				*bkw_data[c-bkw_ind] = *bkw_data[c-bkw_ind] + dw_post(*c);
				if (*bkw_data[c-bkw_ind]>w_max) *bkw_data[c-bkw_ind]=w_max;
			}
		}
	}
}

void RateModulatedConnection::propagate()
{
	// compute the averages
	if ( rate_modulating_group==NULL ) return;

	rate_estimate *= rate_estimate_decay_mul;
	rate_estimate += 1.0*rate_modulating_group->get_spikes_immediate()->size()/rate_estimate_tau/rate_modulating_group->get_post_size();
	rate_modulation_mul = (rate_estimate-rate_target)*eta;

	// if ( sys->get_clock()%100000== 0 )
	// 	cout << rate_modulation_mul << endl;
	

	propagate_forward();
	propagate_backward();
}

RateModulatedConnection::~RateModulatedConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}


void RateModulatedConnection::set_modulating_group(SpikingGroup * group)
{
	if ( group->evolve_locally() )
		rate_modulating_group = group;
}
