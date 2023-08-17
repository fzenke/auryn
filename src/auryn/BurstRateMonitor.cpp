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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "BurstRateMonitor.h"

using namespace auryn;

BurstRateMonitor::BurstRateMonitor(SpikingGroup * source, std::string filename, AurynDouble binsize) : Monitor(filename)
{
	init(source,filename,binsize);
}

void BurstRateMonitor::init(SpikingGroup * source, std::string filename, AurynDouble binsize)
{
	auryn::sys->register_device(this);

	src = source;
	scaleconst = 1.0/binsize/src->get_rank_size();
	ssize = (1.0*binsize/auryn_timestep);
	if ( ssize < 1 ) ssize = 1;
	event_counter = 0;
	burst_counter = 0;

	const double default_tau = 16e-3;
	post_trace = src->get_post_trace(default_tau);
	burst_state = src->get_state_vector("_burst_state");

	set_tau(default_tau);
	thr = 1.0+std::exp(-1.0);


	std::stringstream oss;
	oss << "BurstRateMonitor:: Setting binsize " << binsize << "s";
	auryn::logger->msg(oss.str(),NOTIFICATION);

}

BurstRateMonitor::~BurstRateMonitor()
{
}

void BurstRateMonitor::set_tau(double tau)
{
	post_trace->set_timeconstant(tau);
}

void BurstRateMonitor::execute()
{
	if ( src->evolve_locally() ) {

		// loop over all spikes to separate events from bursts
		SpikeContainer::const_iterator spk;
		for ( spk = src->get_spikes_immediate()->begin() ; 
				spk < src->get_spikes_immediate()->end() ; 
				++spk ) {

			const NeuronID s = src->global2rank(*spk);

			// detect first spike in bursts and non-burst spikes 
			if ( post_trace->get(s) < thr ) { 
				++event_counter;
				burst_state->set(s,-1.0);
			} else // detect second spike in burst
			if ( burst_state->get(s) < 0.0 ) { 
				++burst_counter;
				burst_state->set(s,1);
			}
		}


		// record to file every now and then
		if (auryn::sys->get_clock()%ssize==0) {
			const double burst_rate = scaleconst*burst_counter;
			const double event_rate = scaleconst*event_counter;
			burst_counter = 0;
			event_counter = 0;
			// outfile << std::setiosflags(ios::fixed) << std::setprecision(3);
			outfile << auryn::sys->get_time() 
				<< " " << burst_rate 
				<< " " << event_rate << "\n";
		}

		// last_post_val->copy(post_trace);
	}
}

void BurstRateMonitor::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	ar & burst_counter ;
	ar & event_counter ;
}

void BurstRateMonitor::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	ar & burst_counter ;
	ar & event_counter ;
}
