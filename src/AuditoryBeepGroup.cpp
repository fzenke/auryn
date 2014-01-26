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

#include "AuditoryBeepGroup.h"

void AuditoryBeepGroup::init ( AurynFloat duration, AurynFloat interval, AurynFloat rate )
{
	stimulus_duration = duration/dt;
	if ( duration < interval )
		stimulation_period = interval/dt-stimulus_duration;
	else 
		stimulation_period = interval/dt;
	stimulus_active = false;
	next_event = 0;

	rate_off = 1e-9;
	rate_on  = rate;

	stringstream oss;
	oss << "AuditoryBeepGroup:: Set up with stimulus_duration=" 
		<< stimulus_duration 
		<< " and interval=" 
		<< stimulation_period;
	logger->msg(oss.str(),NOTIFICATION);
}

AuditoryBeepGroup::AuditoryBeepGroup(NeuronID n, AurynFloat duration, AurynFloat interval, AurynDouble rate ) : PoissonGroup( n , rate ) 
{
	init(duration, interval, rate );
}

AuditoryBeepGroup::~AuditoryBeepGroup()
{
}

void AuditoryBeepGroup::evolve()
{
	if ( sys->get_clock() >= next_event ) {
		if ( stimulus_active ) {
			stimulus_active = false;
			set_rate(rate_off);
			next_event = sys->get_clock()+stimulation_period; 
		} else {
			stimulus_active = true;
			// add sync spikes
			for ( NeuronID i = 0 ; i < get_rank_size() ; ++i )
				push_spike(i);
			set_rate(rate_on);
			next_event = sys->get_clock()+stimulus_duration; 
		}
	} else {
		PoissonGroup::evolve();
	}
}
