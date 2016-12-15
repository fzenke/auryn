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

#include "MovingBumpGroup.h"

using namespace auryn;

boost::mt19937 MovingBumpGroup::order_gen = boost::mt19937(); 

void MovingBumpGroup::init ( AurynFloat duration, AurynFloat width, std::string outputfile )
{
	set_duration(duration);
	set_interval(0.0);

	set_width(width*get_size());
	set_floor(0.1);

	auryn::logger->parameter("duration", (int)duration);
	next_event = 0;
	stimulus_active = true;

	pos_min = 0.0;
	pos_max = 1.0;

	std::stringstream oss;
	oss << "MovingBumpGroup:: Set up with stimulus_duration=" 
		<< stimulus_duration 
		<< " and width=" 
		<< profile_width;
	auryn::logger->msg(oss.str(),NOTIFICATION);

	if ( !outputfile.empty()  ) {

		tiserfile.open(outputfile.c_str(),std::ios::out);
		if (!tiserfile) {
			std::stringstream oss2;
			oss2 << "MovingBumpGroup:: Can't open output file " << outputfile;
			auryn::logger->msg(oss2.str(),ERROR);
			exit(1);
		}
		tiserfile.setf(std::ios::fixed);
		tiserfile.precision(log(auryn_timestep)/log(10)+1 );
	}


}

MovingBumpGroup::MovingBumpGroup(
		NeuronID n, 
		AurynFloat duration, 
		AurynFloat width, 
		AurynDouble rate, 
		std::string tiserfile
		) : ProfilePoissonGroup( n , rate ) 
{
	init(duration, width, tiserfile );
}

MovingBumpGroup::~MovingBumpGroup()
{
	tiserfile.close();
}

void MovingBumpGroup::set_floor( AurynFloat floor ) 
{
	floor_ = floor;
}

void MovingBumpGroup::set_width( NeuronID width ) 
{
	profile_width = width;
}

void MovingBumpGroup::set_duration( AurynFloat duration ) 
{
	stimulus_duration = duration/auryn_timestep;
}

void MovingBumpGroup::set_interval( AurynFloat interval ) 
{
	stimulus_interval = interval/auryn_timestep;
}


void MovingBumpGroup::evolve()
{
	if ( auryn::sys->get_clock() >= next_event ) {
		if ( stimulus_active && stimulus_interval>0 ) {
			next_event += stimulus_interval;
			stimulus_active = false;
			set_flat_profile();
		} else {
			next_event += stimulus_duration;
			stimulus_active = true;

			boost::uniform_int<> dist(pos_min*get_size(),pos_max*get_size());
			boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(order_gen, dist);

			NeuronID mean = die();
			tiserfile << auryn::sys->get_time() << " " << mean << std::endl;
			set_gaussian_profile(mean, profile_width, floor_);
		} 
	}
	
	ProfilePoissonGroup::evolve();
}
