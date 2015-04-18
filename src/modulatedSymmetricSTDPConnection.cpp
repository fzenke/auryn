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

#include "modulatedSymmetricSTDPConnection.h"

void modulatedSymmetricSTDPConnection::init_mod(const char * modulation_filename, AurynFloat eta, AurynFloat kappa, AurynFloat tau_stdp, AurynWeight maxweight)
{
    
	filetime = 0;
	modulation_file.open(modulation_filename,ios::in);
	if (!modulation_filename) {
	  stringstream oss;
	  oss << "Can't open time series file " << modulation_filename;
	  logger->msg(oss.str(),ERROR);
	  exit(1);
	}

	init(eta, kappa, tau_stdp, maxweight);

}

void modulatedSymmetricSTDPConnection::init(AurynFloat eta, AurynFloat kappa, AurynFloat tau_stdp, AurynWeight maxweight)
{
    
	set_max_weight(maxweight);
	learning_rate = eta;
	target = kappa;
	kappa_fudge = 2*target*tau_stdp;
    gmod = 1;

	stdp_active = true;
	if ( learning_rate == 0 )
		stdp_active = false;

	set_name("modulatedSymmetricSTDPConnection");

	if ( dst->get_post_size() == 0 ) return;

	tr_pre = src->get_pre_trace(tau_stdp);
	tr_post = dst->get_post_trace(tau_stdp);

}

void modulatedSymmetricSTDPConnection::free()
{	modulation_file.close();
}

modulatedSymmetricSTDPConnection::modulatedSymmetricSTDPConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
        const char * modulation_filename, 
		AurynFloat eta, AurynFloat kappa, AurynFloat tau_stdp, 
		AurynWeight maxweight , TransmitterType transmitter) 
: DuplexConnection(source, destination, filename, transmitter)
{
	init_mod(modulation_filename, eta , kappa, tau_stdp, maxweight);
}

modulatedSymmetricSTDPConnection::modulatedSymmetricSTDPConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynFloat eta, AurynFloat kappa, AurynFloat tau_stdp, 
		AurynWeight maxweight , TransmitterType transmitter) 
: DuplexConnection(source, destination, filename, transmitter)
{
	init(eta , kappa, tau_stdp, maxweight);
}

modulatedSymmetricSTDPConnection::modulatedSymmetricSTDPConnection(SpikingGroup * source, NeuronGroup * destination, 
        const char * modulation_filename,
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat eta, AurynFloat kappa, AurynFloat tau_stdp, 
		AurynWeight maxweight , TransmitterType transmitter, string name) 
: DuplexConnection(source, destination, weight, sparseness, transmitter, name)
{
	init_mod(modulation_filename, eta , kappa, tau_stdp, maxweight);
}

modulatedSymmetricSTDPConnection::modulatedSymmetricSTDPConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat eta, AurynFloat kappa, AurynFloat tau_stdp, 
		AurynWeight maxweight , TransmitterType transmitter, string name) 
: DuplexConnection(source, destination, weight, sparseness, transmitter, name)
{
	init(eta , kappa, tau_stdp, maxweight);
}



modulatedSymmetricSTDPConnection::~modulatedSymmetricSTDPConnection()
{
	free();
}

inline AurynWeight modulatedSymmetricSTDPConnection::dw_pre(NeuronID post)
{
	if (stdp_active) {
		double dw = gmod*learning_rate*(tr_post->get(post)-kappa_fudge);
		return dw;
	}
	else return 0.;
}

inline AurynWeight modulatedSymmetricSTDPConnection::dw_post(NeuronID pre)
{
	if (stdp_active) {
		double dw = gmod*learning_rate*tr_pre->get(pre);
		return dw;
	}
	else return 0.;
}

inline void modulatedSymmetricSTDPConnection::propagate_forward()
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
			// dst->tadd( *c , value , transmitter );
			transmit( *c, value );
			NeuronID translated_spike = dst->global2rank(*c);
			data[c-ind] += dw_pre(translated_spike);
		}
	}
}

inline void modulatedSymmetricSTDPConnection::propagate_backward()
{
	NeuronID * ind = bkw->get_row_begin(0); // first element of index array
	AurynWeight ** data = bkw->get_data_begin();
	SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
	for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
			spike != spikes_end ; ++spike ) {
		for (NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {

			#ifdef CODE_ACTIVATE_PREFETCHING_INTRINSICS
			_mm_prefetch((const char *)data[c-ind+2],  _MM_HINT_NTA);
			#endif
			
			*data[c-ind] += dw_post(*c);
//			if (*data[c-ind]>w_max) {
//				*data[c-ind]=w_max;
//			}
		}
	}
}

void modulatedSymmetricSTDPConnection::propagate()
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

