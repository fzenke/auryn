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

#include "PairInteractionConnection.h"

using namespace auryn;

void PairInteractionConnection::init(AurynWeight maxw)
{
	last_spike_pre = new AurynTime[get_m_rows()];
	last_spike_post = new AurynTime[get_n_cols()];

	for ( unsigned int i = 0 ; i < get_m_rows() ; ++i )
		last_spike_pre[i] = -1; // set this to end of range (so spike is infinitely in the future -- might cause problems without ffast-math

	for ( unsigned int i = 0 ; i < get_n_cols() ; ++i )
		last_spike_post[i] = -1;

	window_pre_post = new AurynFloat[WINDOW_MAX_SIZE];
	window_post_pre = new AurynFloat[WINDOW_MAX_SIZE];

	// initialize window with standard exponential 20ms time constant
	
	set_exponential_window();


	// TODO write proper init for these variables
	stdp_active = true;
	w_max = maxw;

	set_name("PairInteractionConnection");
}

void PairInteractionConnection::free()
{
	delete last_spike_pre;
	delete last_spike_post;

	delete window_pre_post;
	delete window_post_pre;
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
	if ( stdp_active ) {
		if ( diff >= WINDOW_MAX_SIZE ) diff = WINDOW_MAX_SIZE-1;
		double dw = window_post_pre[diff];
		return dw;
	}
	else return 0.;
}

inline AurynWeight PairInteractionConnection::dw_bkw(NeuronID pre)
{
	AurynTime diff = auryn::sys->get_clock()-last_spike_pre[pre];
	if ( stdp_active ) {
		if ( diff >= WINDOW_MAX_SIZE ) diff = WINDOW_MAX_SIZE-1;
		double dw = window_pre_post[diff];
		return dw;
	}
	else return 0.;
}

inline void PairInteractionConnection::propagate_forward()
{
	NeuronID * ind = w->get_row_begin(0); // first element of index array
	AurynWeight * data = w->get_data_begin();
	AurynWeight value;
	SpikeContainer::const_iterator spikes_end = src->get_spikes()->end();
	// process spikes
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != spikes_end ; ++spike ) {
		for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) {
			value = data[c-ind]; 
			//dst->tadd( *c , value , transmitter );
            transmit( *c, value );
            //if (data[c-ind]>0 && data[c-ind]<w_max); //Not sure this is correct: updates only if the given weight is in [0, wmax]. Why?
			data[c-ind] += dw_fwd(*c);
        }
		// update pre_trace
		last_spike_pre[*spike] = auryn::sys->get_clock();
	}
}

inline void PairInteractionConnection::propagate_backward()
{
	NeuronID * ind = bkw->get_row_begin(0); // first element of index array
	AurynWeight ** data = bkw->get_data_begin();
	SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
	for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
			spike != spikes_end ; ++spike ) {
		for (NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {
			if (*data[c-ind]<w_max)
			  *data[c-ind] += dw_bkw(*c);
		}
		// update post trace
		last_spike_post[*spike] = auryn::sys->get_clock();
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
	for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
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

	if ( size > 2*WINDOW_MAX_SIZE )
		auryn::logger->msg("PairInteractionConnection:: STDP window too large ... truncating!",WARNING);

	if ( auryn_timestep < timebinsize )
		auryn::logger->msg("PairInteractionConnection:: Timebinning of loaded STDP window is different from simulator timestep.",WARNING);

	double sum_pre_post = 0 ;
	double sum_post_pre = 0 ;

	// read window file line-by-line
	while ( infile.getline (buffer,256)  )
	{
		sscanf (buffer,"%f %f",&time,&value);
		if ( abs(time) < WINDOW_MAX_SIZE*auryn_timestep ) {
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

	// for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
	// 	std::cout << std::ifstream << window_pre_post[i] << std::endl;
	// }
	// for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
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
	for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
		window_pre_post[i] = Aplus/tau_plus*exp(-i*auryn_timestep/tau_plus);
	}

	for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
		window_post_pre[i] = Aminus/tau_minus*exp(-i*auryn_timestep/tau_minus);
	}

	// zero floor terms 
	set_floor_terms();
}

void PairInteractionConnection::set_floor_terms( double pre_post, double post_pre ) 
{
	window_pre_post[WINDOW_MAX_SIZE-1] = pre_post;
	window_post_pre[WINDOW_MAX_SIZE-1] = post_pre;
}
