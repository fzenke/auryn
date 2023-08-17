/* 
* Copyright 2014-2023 Friedemann Zenke
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

#include "PairInteractionConnection.h"

using namespace auryn;

void PairInteractionConnection::init(AurynWeight maxw)
{
	if ( dst->get_post_size() == 0 ) { 
		logger->debug("PairInteractionConnection:: Init bypass because post has no neurons");
		return; // avoids to run this code on silent nodes with zero post neurons.
	}
	logger->debug("PairInteractionConnection:: Init connection");
	logger->parameter("m",get_m_rows());
	logger->parameter("n",get_n_cols());

	last_spike_pre = new AurynTime[src->get_pre_size()];
	last_spike_post = new AurynTime[dst->get_post_size()];

	logger->debug("PairInteractionConnection:: Init last spike arrays");
	for ( unsigned int i = 0 ; i < src->get_pre_size() ; ++i )
		last_spike_pre[i] = -1; // set this to end of range (so spike is infinitely in the future -- might cause problems without ffast-math

	for ( unsigned int i = 0 ; i < dst->get_post_size() ; ++i )
		last_spike_post[i] = -1;

	logger->debug("PairInteractionConnection:: Init STDP window arrays");
	window_pre_post = new AurynFloat[PAIRINTERACTIONCON_WINDOW_MAX_SIZE];
	window_post_pre = new AurynFloat[PAIRINTERACTIONCON_WINDOW_MAX_SIZE];

	logger->debug("PairInteractionConnection:: Init STDP window with standard exponential 20ms time constant");
	set_exponential_window();

	// TODO write proper init for these variables
	stdp_active = true;
	w_max = maxw;

	set_name("PairInteractionConnection");
}

void PairInteractionConnection::free()
{
	if ( dst->get_post_size() == 0 ) return; // only free if it was also initialized

	logger->debug("PairInteractionConnection:: Freeing dynamic arrays");

	delete [] last_spike_pre;
	delete [] last_spike_post;

	delete [] window_pre_post;
	delete [] window_post_pre;
}

PairInteractionConnection::PairInteractionConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynWeight maxweight , TransmitterType transmitter) 
: DuplexConnection(source, destination, filename, transmitter)
{
	init(maxweight);
}

PairInteractionConnection::PairInteractionConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynWeight maxweight , TransmitterType transmitter, std::string name) 
: DuplexConnection(source, destination, weight, sparseness, transmitter, name)
{
	init(maxweight);
}

PairInteractionConnection::~PairInteractionConnection()
{
	free();
}

inline AurynWeight PairInteractionConnection::dw_fwd(NeuronID post)
{
	AurynTime diff = auryn::sys->get_clock()-last_spike_post[post];
	if ( diff >= PAIRINTERACTIONCON_WINDOW_MAX_SIZE ) diff = PAIRINTERACTIONCON_WINDOW_MAX_SIZE-1;
	double dw = window_post_pre[diff];
	return dw;
}

inline AurynWeight PairInteractionConnection::dw_bkw(NeuronID pre)
{
	AurynTime diff = auryn::sys->get_clock()-last_spike_pre[pre];
	if ( diff >= PAIRINTERACTIONCON_WINDOW_MAX_SIZE ) diff = PAIRINTERACTIONCON_WINDOW_MAX_SIZE-1;
	double dw = window_pre_post[diff];
	return dw;
}

inline void PairInteractionConnection::propagate_forward()
{
	// Loop over all pre spikes 
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		// Loop over all postsynaptic partners 
		for (const NeuronID * c = w->get_row_begin(*spike) ; 
				c != w->get_row_end(*spike) ; 
				++c ) { // c = post index

			// determines the weight of connection
			AurynWeight * weight = w->get_data_ptr(c); 

			// handle plasticity
			if ( stdp_active ) {
				// performs weight update upon presynaptic spike
			    *weight += dw_fwd(*c);
			    // clips too small weights
			    if ( *weight < get_min_weight() ) *weight = get_min_weight(); 
				else if ( *weight > get_max_weight() ) *weight = get_max_weight();
			}

			// evokes the postsynaptic response 
			transmit( *c , *weight );

			// update pre "trace"
			last_spike_pre[*spike] = auryn::sys->get_clock();
		}
	}
}

inline void PairInteractionConnection::propagate_backward()
{
	if (stdp_active) { 
		SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
		// loop over all post spikes
		for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
				spike != spikes_end ; 
				++spike ) {
			NeuronID translated_spike = dst->global2rank(*spike); 

			// loop over all presynaptic partners
			for (const NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {
				// computes plasticity update
				AurynWeight * weight = bkw->get_data(c); 
				*weight += dw_bkw(*c);

				// clips too large weights
				if (*weight>get_max_weight()) *weight=get_max_weight();
			}
			// update post "trace"
			last_spike_post[translated_spike] = auryn::sys->get_clock();
		}
	}
}

void PairInteractionConnection::propagate()
{
	// propagate
	propagate_forward();
	propagate_backward();
}

void PairInteractionConnection::load_window_from_file( const char * filename , double scale ) 
{
	std::stringstream oss;
	oss << "PairInteractionConnection:: Loading STDP window from " << filename;
	auryn::logger->msg(oss.str(),NOTIFICATION);

	// default window all zeros
	for ( int i = 0 ; i < PAIRINTERACTIONCON_WINDOW_MAX_SIZE ; ++i ) {
		window_pre_post[i] = 0;
		window_post_pre[i] = 0;
	}

	std::ifstream infile (filename);
	if (!infile) {
		std::stringstream oes;
		oes << "Can't open input file " << filename;
		auryn::logger->msg(oes.str(),ERROR);
		return;
	}

	unsigned int size;
	float timebinsize;
	float value;
	float time;
	unsigned int count = 0;

	char buffer[256];
	infile.getline (buffer,256); 
	sscanf (buffer,"# %u %f",&size,&timebinsize);

	if ( size > 2*PAIRINTERACTIONCON_WINDOW_MAX_SIZE )
		auryn::logger->msg("PairInteractionConnection:: STDP window too large ... truncating!",WARNING);

	if ( auryn_timestep < timebinsize )
		auryn::logger->msg("PairInteractionConnection:: Timebinning of loaded STDP window is different from simulator timestep.",WARNING);

	double sum_pre_post = 0 ;
	double sum_post_pre = 0 ;

	// read window file line-by-line
	while ( infile.getline (buffer,256)  )
	{
		sscanf (buffer,"%f %f",&time,&value);
		if ( abs(time) < PAIRINTERACTIONCON_WINDOW_MAX_SIZE*auryn_timestep ) {
			NeuronID start;
			if ( time < 0  ) {
				start = -(time+auryn_timestep/2)/auryn_timestep; // plus element is for correct rounding
				window_post_pre[start] = scale*value;
				sum_post_pre += scale*value;
			} else {
				start = (time+auryn_timestep/2)/auryn_timestep; 
				window_pre_post[start] = scale*value;
				sum_pre_post += scale*value;
			}
		}
		count++;
	}

	// for ( int i = 0 ; i < PAIRINTERACTIONCON_WINDOW_MAX_SIZE ; ++i ) {
	// 	std::cout << std::ifstream << window_pre_post[i] << std::endl;
	// }
	// for ( int i = 0 ; i < PAIRINTERACTIONCON_WINDOW_MAX_SIZE ; ++i ) {
	// 	std::cout << std::ifstream << window_post_pre[i] << std::endl;
	// }


	oss.str("");
	oss << "PairInteractionConnection:: sum_pre_post=" 
		<< std::scientific
		<< sum_pre_post 
		<< " sum_post_pre=" 
		<< sum_post_pre;
	auryn::logger->msg(oss.str(),NOTIFICATION);

	infile.close();

}

void PairInteractionConnection::set_exponential_window ( double Aplus, double tau_plus, double Aminus, double tau_minus) 
{
	for ( int i = 0 ; i < PAIRINTERACTIONCON_WINDOW_MAX_SIZE ; ++i ) {
		window_pre_post[i] = Aplus/tau_plus*exp(-i*auryn_timestep/tau_plus);
	}

	for ( int i = 0 ; i < PAIRINTERACTIONCON_WINDOW_MAX_SIZE ; ++i ) {
		window_post_pre[i] = Aminus/tau_minus*exp(-i*auryn_timestep/tau_minus);
	}

	// zero floor terms 
	set_floor_terms(0.0, 0.0);
}

void PairInteractionConnection::set_floor_terms( double pre_post, double post_pre ) 
{
	window_pre_post[PAIRINTERACTIONCON_WINDOW_MAX_SIZE-1] = pre_post;
	window_post_pre[PAIRINTERACTIONCON_WINDOW_MAX_SIZE-1] = post_pre;
}
