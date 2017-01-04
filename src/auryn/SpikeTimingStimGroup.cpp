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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "SpikeTimingStimGroup.h"

using namespace auryn;


SpikeTimingStimGroup::SpikeTimingStimGroup(NeuronID n, std::string filename, std::string stimfile, StimulusGroupModeType stimulusmode, AurynFloat timeframe) : StimulusGroup( n, filename, stimfile, stimulusmode, timeframe ) 
{
	init();
}

SpikeTimingStimGroup::SpikeTimingStimGroup(NeuronID n, std::string stimfile, StimulusGroupModeType stimulusmode, AurynFloat timeframe) : StimulusGroup( n, stimfile, stimulusmode, timeframe ) 
{
	init();
}

SpikeTimingStimGroup::~SpikeTimingStimGroup()
{
}

void SpikeTimingStimGroup::init()
{
	refractory_period = 0.0;
	scale = 1.0;
}

void SpikeTimingStimGroup::redraw()
{
	for ( NeuronID i = 0; i < get_rank_size() ; ++i ) {
		ttl[i] = sys->get_clock() + activity[i]/auryn_timestep;
	}
}

void SpikeTimingStimGroup::evolve()
{
	if ( !active ) return;

	if ( stimulus_active  ) { // during active stimulation

		for ( NeuronID i = 0; i < get_rank_size() ; ++i ) {
			// if ( ttl[i] ) { // DEBUG
			// 	std::cout << ttl[i] << " " << sys->get_clock() << std::endl;
			// }
			if ( ttl[i] == sys->get_clock() ) {
				push_spike(i);
			}
		}

	} else { // while stimulation is off
		if ( background_rate ) {
			boost::exponential_distribution<> dist(background_rate);
			boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(poisson_gen, dist);

			while ( bgx < get_rank_size() ) {
				push_spike ( bgx );
				AurynDouble r = die();
				bgx += 1+(NeuronID)(r/auryn_timestep); 
			}
			bgx -= get_rank_size();
		}
	}

	// update stimulus properties
	if ( auryn::sys->get_clock() >= next_action_time ) { // action required
		last_action_time = next_action_time; // store last time before updating next_action_time

		if ( stimuli.size() == 0 ) {
			set_next_action_time(10); // TODO make this a bit smarter at some point -- i.e. could send this to the end of time 
			return;
		}

		write_stimulus_file(auryn_timestep*(auryn::sys->get_clock()));

		if ( stimulus_order == STIMFILE )  {
			AurynDouble t = 0.0;
			int a,i;
			while ( t <= auryn::sys->get_time() ) {
				read_next_stimulus_from_file(t,a,i);
				next_action_time = (AurynTime) (t/auryn_timestep);
				if (a==0) stimulus_active = true; 
					else stimulus_active = false;
				cur_stim_index = i;
				// std::cout << auryn::sys->get_time() << " " << t << " " << a << " " << i << std::endl;
			}
		} else { // we have to generate stimulus times

			if ( stimulus_active ) { // stimulus was active and going inactive now

				set_all( 0.0 ); // turns off currently active stimulus 
				stimulus_active = false ;

				if ( randomintervals ) {
					boost::exponential_distribution<> dist(1./mean_off_period);
					boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(order_gen, dist);
					next_action_time = auryn::sys->get_clock() + (AurynTime)(std::max(0.0,die()+refractory_period)/auryn_timestep);
				} else {
					next_action_time = auryn::sys->get_clock() + (AurynTime)((mean_off_period+refractory_period)/auryn_timestep);
				}
			} else { // stimulus was not active and is going active now
				if ( active && stimuli.size() ) { // the group is active and there are stimuli in the array

					// chooses stimulus according to schema specified in stimulusmode
					double draw, cummulative;
					switch ( stimulus_order ) {
						case RANDOM:
							// TODO make this less greedy 
							// and do not compute this every draw
							draw = order_die();
							cummulative = 0; 
							cur_stim_index = 0;
							// std::cout.precision(5);
							// std::cout << " draw " << draw <<  std::endl;
							for ( unsigned int i = 0 ; i < probabilities.size() ; ++i ) {
								cummulative += probabilities[i];
								// std::cout << cummulative << std::endl;
								if ( draw <= cummulative ) {
									cur_stim_index = i;
									break;
								}
							}
						break;
						case SEQUENTIAL:
							cur_stim_index = (cur_stim_index+1)%stimuli.size();
						break;
						case SEQUENTIAL_REV:
							--cur_stim_index;
							if ( cur_stim_index <= 0 ) 
								cur_stim_index = stimuli.size() - 1 ;
						break;
						case MANUAL:
						default:
						break;
					}

					// sets the activity
					set_active_pattern( cur_stim_index, 1e20 ); // puts default spikes into the not forseeable future
					stimulus_active = true;

					next_action_time = auryn::sys->get_clock() + (AurynTime)(mean_on_period/auryn_timestep);
				}
			}
			write_stimulus_file(auryn_timestep*(auryn::sys->get_clock()+1));
		}
	}
}

