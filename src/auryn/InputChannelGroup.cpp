/* 
* Copyright 2014-2018 Friedemann Zenke
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

#include "InputChannelGroup.h"

using namespace auryn;

boost::mt19937 InputChannelGroup::shared_noise_gen = boost::mt19937(); 
boost::mt19937 InputChannelGroup::rank_noise_gen   = boost::mt19937(); 

void InputChannelGroup::init(AurynDouble  rate, NeuronID chsize )
{
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) {
		lambda = rate;

		channelsize = global2rank(chsize);
		nb_channels = size/chsize;
		offset = 0;

		mean = 0.0;

		timescale = 50e-3;
		set_amplitude(50.0);
		set_stoptime(0);

		std::stringstream oss;
		oss << "InputChannelGroup:: Initializing with " 
			<< nb_channels << " of size " 
			<< channelsize << " ( "
			<< " timescale=" << timescale 
			<< " )";
		auryn::logger->msg(oss.str(),NOTIFICATION);

		seed(); // seed the generators
		// TODO fix problem with copy constructor of uniform_01
		// See http://www.bnikolic.co.uk/blog/cpp-boost-uniform01.html
		shared_noise = new boost::variate_generator<boost::mt19937&, boost::uniform_01<> > ( shared_noise_gen, boost::uniform_01<> () ); // should have the same state on all ranks
		rank_noise = new boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > ( rank_noise_gen, boost::exponential_distribution<>(1.0) ); // should have a different state on each rank

		o = new AurynDouble [nb_channels];
		for ( unsigned int i = 0 ; i < nb_channels ; ++i ) {
			o[i] = 0.0; 
		}

		x = new NeuronID [nb_channels];
		for ( unsigned int i = 0 ; i < nb_channels ; ++i ) {
			const AurynDouble r = (*rank_noise)()/lambda;
			x[i] += (NeuronID)(r/auryn_timestep); 
		}

		oss.str("");
		oss << "InputChannelGroup:: Seeding with " << sys->mpi_rank();
		auryn::logger->msg(oss.str(),NOTIFICATION);
	}
}

InputChannelGroup::InputChannelGroup(NeuronID n, 
		AurynDouble  rate, 
		NeuronID chsize ) : SpikingGroup( n ) 
{
	init(rate, chsize);
}

InputChannelGroup::~InputChannelGroup()
{
	if ( evolve_locally() ) {
		// delete dist;
		delete shared_noise;
		delete rank_noise;
		delete [] x;
		delete [] o;
	}
}

void InputChannelGroup::set_rate(AurynDouble  rate)
{
	lambda = rate;
}

AurynDouble  InputChannelGroup::get_rate()
{
	return lambda;
}


void InputChannelGroup::evolve()
{
	// check if the group has timed out
	if ( tstop && auryn::sys->get_clock() > tstop ) return;

	for ( unsigned int g = 0 ; g < nb_channels ; ++g ) { // loop over channels
		// integrate Ornstein Uhlenbeck process
		o[g] += ( mean - o[g] )*auryn_timestep/timescale;

		// noise increment
		o[g] += 2.0*((AurynDouble)(*shared_noise)()-0.5)*std::sqrt(auryn_timestep/timescale);

		// group rate
		AurynDouble grouprate = amplitude*o[g];  
		if ( grouprate <= 0 ) grouprate = 0.0;

		const AurynFloat epsilon = 1e-3;
		AurynDouble r = (*rank_noise)()/(auryn_timestep*(grouprate+epsilon)); // think before tempering with this! 
		// I already broke the code here once!
		x[g] = (NeuronID)(r); 
		while ( x[g] < channelsize ) {
			push_spike ( g*channelsize + x[g] );
			r = (*rank_noise)()/(auryn_timestep*(grouprate+epsilon));
			x[g] += (NeuronID)(r); 
		}
	}
}

void InputChannelGroup::seed()
{
		shared_noise_gen.seed(sys->get_synced_seed()); 
		rank_noise_gen.seed(sys->get_seed()); 
}

void InputChannelGroup::set_amplitude(AurynDouble amp)
{
	amplitude = amp;
	auryn::logger->parameter("amplitude",amplitude);
}

void InputChannelGroup::set_timescale(AurynDouble scale)
{
	timescale = scale;
	auryn::logger->parameter("timescale",timescale);
}

void InputChannelGroup::set_offset(int off)
{
	offset = off;
	auryn::logger->parameter("offset",offset);
}

void InputChannelGroup::set_stoptime(AurynDouble stoptime)
{
	tstop = stoptime*auryn_timestep;
	auryn::logger->parameter("stoptime",stoptime);
}
