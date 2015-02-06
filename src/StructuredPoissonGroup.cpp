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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "StructuredPoissonGroup.h"

boost::mt19937 StructuredPoissonGroup::interval_gen = boost::mt19937(); 

void StructuredPoissonGroup::init ( AurynFloat duration, AurynFloat mean_interval, NeuronID no, string outputfile )
{
	no_of_stimuli = no;
	stimulus_duration = duration;
	mean_isi = mean_interval;
	stimulus_active = false;
	current_stimulus = 0;
	next_event = mean_isi;

	stringstream oss;
	oss << "StructuredPoissonGroup:: Set up with stimulus_duration=" 
		<< stimulus_duration 
		<< " and mean_isi=" 
		<< mean_isi;
	logger->msg(oss.str(),NOTIFICATION);

	if ( evolve_locally() ) {
		dist = new boost::exponential_distribution<> (1./mean_interval);
		die  = new boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > ( interval_gen, *dist );

		if ( !outputfile.empty() ) 
		{
			tiserfile.open(outputfile.c_str(),ios::out);
			if (!tiserfile) {
			  stringstream oss2;
			  oss2 << "StructuredPoissonGroup:: Can't open output file " << outputfile;
			  logger->msg(oss2.str(),ERROR);
			  exit(1);
			}
			tiserfile.setf(ios::fixed);
			tiserfile.precision(log(dt)/log(10)+1 );
		}
	}
}

StructuredPoissonGroup::StructuredPoissonGroup(NeuronID n, AurynFloat duration, AurynFloat interval, NeuronID stimuli,
		AurynDouble rate , string tiserfile ) : PoissonGroup( n , rate ) 
{
	init(duration, interval, stimuli, tiserfile );
}

StructuredPoissonGroup::~StructuredPoissonGroup()
{
	if ( evolve_locally() ) {
		delete dist;
		delete die;
		tiserfile.close();
	}
}

void StructuredPoissonGroup::evolve()
{
	if ( sys->get_clock() >= next_event ) {
		if ( stimulus_active ) {
			stimulus_active = false;
			seed(sys->get_clock());
			next_event = sys->get_clock()+((AurynTime)((*die)()/dt));
		} else {
			stimulus_active = true;
			current_stimulus = (current_stimulus+1)%no_of_stimuli;
			x = 0;
			tiserfile << sys->get_time() << " " << current_stimulus << endl;
			seed(current_stimulus);
			next_event = sys->get_clock()+(AurynTime)(stimulus_duration/dt); 
		}
	}
	PoissonGroup::evolve();
}
