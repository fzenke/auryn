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

#include "MinimalTripletConnection.h"

using namespace auryn;

void MinimalTripletConnection::init(AurynFloat eta, AurynFloat maxweight)
{
	if ( dst->get_post_size() == 0 ) return; // avoids to run this code on silent nodes with zero post neurons.

	/* Initialization of plasticity parameters. */
	A3_plus  = 8.0e-3;
	A2_minus = 3.5e-3;
	A2_plus  = 5.3e-3;

	A3_plus  *= eta;
	A2_minus *= eta;
	A2_plus  *= eta;

	tau_plus  = 16.8e-3;
	tau_minus = 33.7e-3;
	tau_long  = 40e-3;

	/* Initialization of presynaptic traces */
	tr_pre = src->get_pre_trace(tau_plus);

	/* Initialization of postsynaptic traces */
	tr_post = dst->get_post_trace(tau_minus);
	tr_post2 = dst->get_post_trace(tau_long);

	/* Set min/max weight values. */
	set_min_weight(0.0);
	set_max_weight(maxweight);

	stdp_active = true;

}

void MinimalTripletConnection::init_shortcuts() 
{
	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

	fwd_ind = w->get_row_begin(0); 
	fwd_data = w->get_data_begin();

	bkw_ind = bkw->get_row_begin(0); 
	bkw_data = bkw->get_data_begin();
}

void MinimalTripletConnection::finalize() {
	DuplexConnection::finalize();
	init_shortcuts();
}

void MinimalTripletConnection::free()
{
}

MinimalTripletConnection::MinimalTripletConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) : DuplexConnection(source, destination, transmitter)
{
}

MinimalTripletConnection::MinimalTripletConnection(SpikingGroup * source, NeuronGroup * destination, 
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

MinimalTripletConnection::MinimalTripletConnection(
		SpikingGroup * source, 
		NeuronGroup * destination, 
		AurynWeight weight, 
		AurynFloat sparseness, 
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
		set_name("MinimalTripletConnection");
	init_shortcuts();
}

MinimalTripletConnection::~MinimalTripletConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}

/*! This function implements what happens to synapes transmitting a 
 *  spike to neuron 'post'. */
AurynWeight MinimalTripletConnection::dw_pre(NeuronID post)
{
	// translate post id to local id on rank: translated_spike
	NeuronID translated_spike = dst->global2rank(post); 
	AurynDouble dw = -A2_minus*(tr_post->get(translated_spike));
	return dw;
}

/*! This function implements what happens to synapes experiencing a 
 *  backpropagating action potential from neuron 'pre'. */
AurynWeight MinimalTripletConnection::dw_post(NeuronID pre, NeuronID post)
{
	// at this point post was already translated to a local id in 
	// the propagate_backward function below.
	AurynDouble dw = (A2_plus+A3_plus*tr_post2->get(post))*tr_pre->get(pre);
	return dw;
}


void MinimalTripletConnection::propagate_forward()
{
	// loop over all spikes
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		// loop over all postsynaptic partners
		for (const NeuronID * c = w->get_row_begin(*spike) ; 
				c != w->get_row_end(*spike) ; 
				++c ) { // c = post index

			// transmit signal to target at postsynaptic neuron
			AurynWeight * weight = w->get_data_ptr(c); 
			transmit( *c , *weight );

			// handle plasticity
			if ( stdp_active ) {
				// performs weight update
			    *weight += dw_pre(*c);

			    // clips too small weights
			    if ( *weight < get_min_weight() ) 
					*weight = get_min_weight();
			}
		}
	}
}

void MinimalTripletConnection::propagate_backward()
{
	if (stdp_active) { 
		SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
		// loop over all spikes
		for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
				spike != spikes_end ; 
				++spike ) {
			// Since we need the local id of the postsynaptic neuron that spiked 
			// multiple times, we translate it here:
			NeuronID translated_spike = dst->global2rank(*spike); 

			// loop over all presynaptic partners
			for (const NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {

				#ifdef CODE_ACTIVATE_PREFETCHING_INTRINSICS
				// prefetches next memory cells to reduce number of last-level cache misses
				_mm_prefetch((const char *)bkw_data[c-bkw_ind+2],  _MM_HINT_NTA);
				#endif

				// computes plasticity update
				AurynWeight * weight = bkw->get_data(c); 
				*weight += dw_post(*c,translated_spike);

				// clips too large weights
				if (*weight>get_max_weight()) *weight=get_max_weight();
			}
		}
	}
}

void MinimalTripletConnection::propagate()
{
	propagate_forward();
	propagate_backward();
}

void MinimalTripletConnection::evolve()
{
}


