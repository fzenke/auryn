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
*/

#include "modulatedPairInteractionConnection.h"


boost::mt19937 modulatedPairInteractionConnection::gen = boost::mt19937();

void modulatedPairInteractionConnection::check_modulation_file(const char * modulation_filename){
	modulation_file.open(modulation_filename,ios::in);
	if (!modulation_filename) {
	  stringstream oss;
	  oss << "Can't open time series file " << modulation_filename;
	  logger->msg(oss.str(),ERROR);
	  exit(1);
	}
}

void modulatedPairInteractionConnection::init(AurynWeight maxw)
{
    gmod = 1;
	filetime = 0;
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

	seed(61093*communicator->rank());
	dist = new boost::uniform_int<int> (0, 1);
	die  = new boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > ( gen, *dist );

	set_name("modulatedPairInteractionConnection");
}

void modulatedPairInteractionConnection::free()
{
	delete last_spike_pre;
	delete last_spike_post;

	delete window_pre_post;
	delete window_post_pre;

	delete dist;
	delete die;
}

modulatedPairInteractionConnection::modulatedPairInteractionConnection(SpikingGroup * source, NeuronGroup * destination,  const char * modulation_filename, 
		const char * filename, AurynWeight maxweight , TransmitterType transmitter) 
: DuplexConnection(source, destination, filename, transmitter)
{
    check_modulation_file(modulation_filename);
	init(maxweight);
}

modulatedPairInteractionConnection::modulatedPairInteractionConnection(SpikingGroup * source, NeuronGroup * destination,  const char * modulation_filename, 
		AurynWeight weight, AurynFloat sparseness, AurynWeight maxweight , TransmitterType transmitter, string name) 
: DuplexConnection(source, destination, weight, sparseness, transmitter, name)
{
    check_modulation_file(modulation_filename);
	init(maxweight);
}

modulatedPairInteractionConnection::~modulatedPairInteractionConnection()
{
	free();
}

inline AurynWeight modulatedPairInteractionConnection::dw_fwd(NeuronID post)
{
	AurynTime diff = sys->get_clock()-last_spike_post[post];
	if ( stdp_active ) {
		if ( diff >= WINDOW_MAX_SIZE ) diff = WINDOW_MAX_SIZE-1;
		double dw = gmod*window_post_pre[diff];
		return dw;
	}
	else return 0.;
}

inline AurynWeight modulatedPairInteractionConnection::dw_bkw(NeuronID pre)
{
	AurynTime diff = sys->get_clock()-last_spike_pre[pre];
	if ( stdp_active ) {
		if ( diff >= WINDOW_MAX_SIZE ) diff = WINDOW_MAX_SIZE-1;
		double dw = gmod*window_pre_post[diff];
		return dw;
	}
	else return 0.;
}

inline void modulatedPairInteractionConnection::propagate_forward()
{
	NeuronID * ind = w->get_row_begin(0); // first element of index array
	AurynWeight * data = w->get_data_begin();
	AurynWeight value;
	TransmitterType transmitter = get_transmitter();
	SpikeContainer::const_iterator spikes_end = src->get_spikes()->end();
	// process spikes
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != spikes_end ; ++spike ) {
		for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) {
			value = data[c-ind]; 
            transmit( *c, value);
			data[c-ind] += dw_fwd(*c);
        }
		// update pre_trace
		last_spike_pre[*spike] = sys->get_clock();
	}
}

inline void modulatedPairInteractionConnection::propagate_backward()
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
		last_spike_post[*spike] = sys->get_clock();
	}
}

void modulatedPairInteractionConnection::propagate()
{
	// propagate
    if ( dst->evolve_locally() ) {

	char buffer[25600];
	string line;
	while( !modulation_file.eof() && filetime < sys->get_clock() ) {
		line.clear();
		modulation_file.getline (buffer,25599);
		line = buffer;
		if (line[0] == '#') continue;
		stringstream iss (line);
		double time;
		iss >> time;
		filetime = time/dt;

        AurynFloat cur;
        iss >> cur ;
        mods = newmods;
        newmods = cur;
		}
	}
        
    gmod = mods;
	propagate_forward();
	propagate_backward();
}

void modulatedPairInteractionConnection::load_window_from_file( const char * filename , double scale ) 
{

	stringstream oss;
	oss << "modulatedPairInteractionConnection:: Loading STDP window from " << filename;
	logger->msg(oss.str(),NOTIFICATION);

	// default window all zeros
	for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
		window_pre_post[i] = 0;
		window_post_pre[i] = 0;
	}

	ifstream infile (filename);
	if (!infile) {
		stringstream oes;
		oes << "Can't open input file " << filename;
		logger->msg(oes.str(),ERROR);
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
		logger->msg("modulatedPairInteractionConnection:: STDP window too large ... truncating!",WARNING);

	if ( dt < timebinsize )
		logger->msg("modulatedPairInteractionConnection:: Timebinning of loaded STDP window is different from simulator timestep.",WARNING);

	double sum_pre_post = 0 ;
	double sum_post_pre = 0 ;

	// read window file line-by-line
	while ( infile.getline (buffer,256)  )
	{
		sscanf (buffer,"%f %f",&time,&value);
		if ( abs(time) < WINDOW_MAX_SIZE*dt ) {
			NeuronID start;
			if ( time < 0  ) {
				start = -(time+dt/2)/dt; // plus element is for correct rounding
				window_post_pre[start] = scale*value;
				sum_post_pre += scale*value;
			} else {
				start = (time+dt/2)/dt; 
				window_pre_post[start] = scale*value;
				sum_pre_post += scale*value;
			}
		}
		count++;
	}

	// for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
	// 	cout << scientific << window_pre_post[i] << endl;
	// }
	// for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
	// 	cout << scientific << window_post_pre[i] << endl;
	// }


	oss.str("");
	oss << "modulatedPairInteractionConnection:: sum_pre_post=" 
		<< scientific 
		<< sum_pre_post 
		<< " sum_post_pre=" 
		<< sum_post_pre;
	logger->msg(oss.str(),NOTIFICATION);

	infile.close();

}

void modulatedPairInteractionConnection::set_exponential_window ( double Aplus, double tau_plus, double Aminus, double tau_minus) 
{
	for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
		window_pre_post[i] = Aplus/tau_plus*exp(-i*dt/tau_plus);
	}

	for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
		window_post_pre[i] = Aminus/tau_minus*exp(-i*dt/tau_minus);
	}

	// zero floor terms 
	set_floor_terms();
}

void modulatedPairInteractionConnection::set_box_window ( double Aplus, double tau_plus, double Aminus, double tau_minus) 
{
	for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
        if (i*dt<tau_plus){
		window_pre_post[i] = Aplus;
        } else {
		window_pre_post[i] = 0;
        }
	}

	for ( int i = 0 ; i < WINDOW_MAX_SIZE ; ++i ) {
        if (i*dt<tau_minus){
		window_post_pre[i] = Aminus;
        } else {
		window_post_pre[i] = 0;
        }
	}

	// zero floor terms 
	set_floor_terms(0, 0);
}

void modulatedPairInteractionConnection::set_floor_terms( double pre_post, double post_pre ) 
{
	window_pre_post[WINDOW_MAX_SIZE-1] = pre_post;
	window_post_pre[WINDOW_MAX_SIZE-1] = post_pre;
}
