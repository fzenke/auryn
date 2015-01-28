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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "LPTripletConnection.h"

void LPTripletConnection::init(AurynFloat tau_hom, AurynFloat eta, AurynFloat kappa, AurynFloat maxweight)
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

	tau_lp = 120;
	timestep_lp = 1e-3*tau_lp/dt;
	delta_lp = 1.0*timestep_lp/tau_lp*dt;


	// Set number of synaptic states
	w->set_num_synapse_states(2);

	// copy all the elements from z=0 to z=1
	w->state_set_all(w->get_state_begin(1),0.0);
	w->state_add(w->get_state_begin(0),w->get_state_begin(1));

	/* Define temporary state vectors */
	// FIXME have to make sure they size is adapted upon resizing of datasize
	temp_state = new AurynWeight[w->get_statesize()];

	// Run finalize again to rebuild backward matrix
	finalize(); 
}

void LPTripletConnection::finalize() {
	// will compute backward matrix on the new elements/data vector of the w
	DuplexConnection::finalize();
}

void LPTripletConnection::free()
{
	delete [] temp_state;
}

LPTripletConnection::LPTripletConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) : DuplexConnection(source, destination, transmitter)
{
}

LPTripletConnection::LPTripletConnection(SpikingGroup * source, NeuronGroup * destination, 
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
}

LPTripletConnection::LPTripletConnection(SpikingGroup * source, NeuronGroup * destination, 
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
		set_name("LPTripletConnection");
}

LPTripletConnection::~LPTripletConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}

void LPTripletConnection::set_hom_trace(AurynFloat freq)
{
	if ( dst->get_post_size() > 0 ) 
		tr_post_hom->set_all(freq*tr_post_hom->get_tau());
}


AurynWeight LPTripletConnection::get_hom(NeuronID i)
{
	return pow(tr_post_hom->get(i),2);
}


/*! This function implements what happens to synapes transmitting a 
 *  spike to neuron 'post'. */
AurynWeight LPTripletConnection::dw_pre(NeuronID post)
{
	// translate post id to local id on rank: translated_spike
	NeuronID translated_spike = dst->global2rank(post); 
	AurynDouble dw = -hom_fudge*(tr_post->get(translated_spike)*get_hom(translated_spike));
	return dw;
}

/*! This function implements what happens to synapes experiencing a 
 *  backpropagating action potential from neuron 'pre'. */
AurynWeight LPTripletConnection::dw_post(NeuronID pre, NeuronID post)
{
	// at this point post was already translated to a local id in 
	// the propagate_backward function below.
	AurynDouble dw = A3_plus*tr_pre->get(pre)*tr_post2->get(post);
	return dw;
}


void LPTripletConnection::propagate_forward()
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
				// transmit signal to target at postsynaptic neuron
				AurynWeight * lpweight = w->get_data_ptr(c,1); 

				// performs weight update
			    *lpweight += dw_pre(*c);

			    // clips too small weights
			    if ( *lpweight < get_min_weight() ) 
					*lpweight = get_min_weight();
			}
		}
	}
}

void LPTripletConnection::propagate_backward()
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
				// define shortcut
				AurynWeight * current_element = bkw->get_data(c);

				#ifdef CODE_ACTIVATE_PREFETCHING_INTRINSICS
				// prefetches next memory cells to reduce number of last-level cache misses
				_mm_prefetch((const char *)(current_element+2),  _MM_HINT_NTA);
				#endif

				// computes plasticity update
				*current_element = *current_element + dw_post(*c,translated_spike);

				// clips too large weights
				if (*current_element>get_max_weight()) *current_element=get_max_weight();
			}
		}
	}
}

void LPTripletConnection::propagate()
{
	propagate_forward();
	propagate_backward();
}

void LPTripletConnection::evolve()
{
	if ( sys->get_clock()%timestep_lp == 0 && stdp_active ) {
		AurynWeight * lpwval = w->get_data_begin(0);
		AurynWeight * wval   = w->get_data_begin(1);
		w->state_sub(wval,lpwval,temp_state);
		w->state_saxpy(delta_lp,temp_state,lpwval);
	}
}


