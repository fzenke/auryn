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
*/

#include "TripletScalingConnection.h"

using namespace auryn;

void TripletScalingConnection::init(AurynFloat tau_hom, AurynFloat eta, AurynFloat kappa, AurynFloat beta, AurynFloat maxweight)
{
	if ( dst->get_post_size() == 0 ) return;

	A3_plus = 6.5e-3;
	A3_plus *= eta;

	tau_plus  = 16.8e-3;
	tau_minus = 33.7e-3;
	tau_long  = 114e-3;

	tau_homeostatic = tau_hom;
	set_beta(beta);

	tr_pre = src->get_pre_trace(tau_plus);
	tr_post = dst->get_post_trace(tau_minus);
	tr_post2 = dst->get_post_trace(tau_long);
	tr_post_hom = dst->get_post_trace(tau_hom);

	target_rate = kappa;
	hom_fudge = A3_plus*(tau_plus*tau_long)/(tau_minus)*target_rate;
	logger->parameter("hom_fudge",hom_fudge);

	w_min = 0.0;
	w_max = maxweight;

	stdp_active = true;

	set_name("TripletScalingConnection");
}

void TripletScalingConnection::init_shortcuts() 
{
	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

	fwd_ind = w->get_row_begin(0); 
	fwd_data = w->get_data_begin();

	bkw_ind = bkw->get_row_begin(0); 
	bkw_data = bkw->get_data_begin();
}

void TripletScalingConnection::finalize() {
	DuplexConnection::finalize();
	init_shortcuts();
}

void TripletScalingConnection::free()
{
}

TripletScalingConnection::TripletScalingConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) : DuplexConnection(source, destination, transmitter)
{
}

TripletScalingConnection::TripletScalingConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynFloat tau_hom, 
		AurynFloat eta, 
		AurynFloat kappa, 
		AurynFloat beta,
		AurynFloat maxweight , 
		TransmitterType transmitter) 
: DuplexConnection(source, 
		destination, 
		filename, 
		transmitter)
{
	init(tau_hom, eta, kappa, beta, maxweight);
	init_shortcuts();
}

TripletScalingConnection::TripletScalingConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat tau_hom, 
		AurynFloat eta, 
		AurynFloat kappa, 
		AurynFloat beta,
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
	init(tau_hom, eta, kappa, beta, maxweight);
	init_shortcuts();
}

TripletScalingConnection::~TripletScalingConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}

void TripletScalingConnection::set_hom_trace(AurynFloat freq)
{
	if ( dst->get_post_size() > 0 ) 
		tr_post_hom->set_all(freq*tr_post_hom->get_tau());
}


AurynWeight TripletScalingConnection::get_hom(NeuronID i)
{
	return pow(tr_post_hom->get(i),2);
}


AurynWeight TripletScalingConnection::dw_pre(NeuronID post)
{
	NeuronID translated_spike = dst->global2rank(post); // only to be used for post traces
	AurynDouble dw = hom_fudge*(tr_post->get(translated_spike));
	// cout << "pre" << dw << endl;
	return dw;
}

AurynWeight TripletScalingConnection::dw_post(NeuronID pre, NeuronID post)
{
	// post translation is done in loop below
	AurynDouble dw = A3_plus*tr_pre->get(pre)*tr_post2->get(post);
	// cout << "post" << dw << endl;
	return dw;
}


void TripletScalingConnection::propagate_forward()
{
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) { // c = post index
			AurynWeight value = fwd_data[c-fwd_ind]; 
			transmit( *c , value );
			if ( stdp_active ) {
			  fwd_data[c-fwd_ind] -= dw_pre(*c);
			  if ( fwd_data[c-fwd_ind] < w_min ) 
				fwd_data[c-fwd_ind] = w_min;
			}
		}
		// update pre_trace
		// tr_pre->inc(*spike);
	}
}

void TripletScalingConnection::propagate_backward()
{
	SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
	// process spikes
	for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
			spike != spikes_end ; ++spike ) {
		NeuronID translated_spike = dst->global2rank(*spike); // only to be used for post traces
		if ( stdp_active ) {
			for (NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {
				*bkw_data[c-bkw_ind] = *bkw_data[c-bkw_ind] + dw_post(*c,translated_spike);
				if (*bkw_data[c-bkw_ind]>w_max) *bkw_data[c-bkw_ind]=w_max;
			}
		}
	}
}

void TripletScalingConnection::propagate()
{
	propagate_forward();
	propagate_backward();
}

void TripletScalingConnection::evolve()
{
	evolve_scaling();
}

void TripletScalingConnection::set_min_weight(AurynWeight min)
{	
	w_min = min;
}

void TripletScalingConnection::set_max_weight(AurynWeight max)
{	
	w_max = max;
}

AurynWeight TripletScalingConnection::get_wmin()
{
	return w_min;
}

void TripletScalingConnection::evolve_scaling() 
{
	// if ( !stdp_active ) return;
	// NeuronID i = sys->get_clock()%scal_timestep;
	// while ( i < dst->get_rank_size() ) {
	// 	const AurynFloat regulator = 1.0-pow(tr_post_hom->normalized_get(i)/target_rate,3);
	// 	NeuronID neuron = dst->rank2global(i);
	// 	for (NeuronID * c = bkw->get_row_begin(neuron) ; c != bkw->get_row_end(neuron) ; ++c ) {
	// 		*bkw_data[c-bkw_ind] += scal_mul*regulator*(*bkw_data[c-bkw_ind]);
	// 	}
	// 	i += scal_timestep;
	// }
	// NeuronID i = sys->get_clock()%scal_timestep;
	// while ( i < dst->get_size() ) {
	// 	for (NeuronID * j = fwd->get_row_begin(i) ; 
	// 			j != fwd->get_row_end(i) ; 
	// 			++j ) { 
	// 		AurynWeight * cor = fwd->get_value_ptr(j);
	// 		AurynFloat diff = target_rate-tr_post_hom->normalized_get(dst->global2rank(*j));
	// 		*cor += (TRIPLETSCALINGCONNECTION_EULERUPGRADE_STEP)*diff*(*cor);
	// 	}
	// 	i += scal_timestep;
	// }
	
	if ( !stdp_active ) {
		return;
	}

	const NeuronID offset = sys->get_clock()%scal_timestep;
	for ( NeuronID pre = offset ; pre < src->get_pre_size() ; pre += scal_timestep  ) {
		for (NeuronID * post = fwd->get_row_begin(pre) ; 
				post != fwd->get_row_end(pre) ; 
				++post ) { 
			AurynWeight * tmp = fwd->get_value_ptr(post);
			const NeuronID translated = dst->global2rank(*post);
			const AurynFloat regulator = 1.0-pow(tr_post_hom->normalized_get(translated)/target_rate,3);
			*tmp += scal_mul*regulator*(*tmp);
		}
	}
}

void TripletScalingConnection::set_beta(AurynFloat beta)
{
	scal_beta = beta;
	// scal_beta *= A3_plus*tau_plus*tau_long;
	logger->parameter("beta",beta);

	scal_timestep = 1000;
	scal_mul = scal_timestep*auryn_timestep/scal_beta;

	logger->parameter("scaling_timestep",(int)scal_timestep);
	logger->parameter("scal_mul",scal_mul);
}
