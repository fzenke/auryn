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

#include "NormalStimulator.h"

using namespace auryn;

boost::mt19937 NormalStimulator::gen = boost::mt19937();

NormalStimulator::NormalStimulator(NeuronGroup * target, AurynWeight sigma, std::string target_state ) : Monitor( )
{
	init(target, sigma, target_state);
}


void NormalStimulator::init( NeuronGroup * target, AurynWeight sigma, std::string target_state )
{
	auryn::sys->register_monitor(this);
	dst = target;

	set_target_state(target_state);

	normal_sigma = sigma;


	std::stringstream oss;
	oss << std::scientific << "NormalStimulator:: initializing with mean " << get_lambda();
	auryn::logger->msg(oss.str(),NOTIFICATION);

	seed(61093*auryn::communicator->rank());
	dist = new boost::normal_distribution<float> (0.0, get_lambda());
	die  = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<float> > ( gen, *dist );
}

void NormalStimulator::free( ) 
{
	delete dist;
	delete die;
}


NormalStimulator::~NormalStimulator()
{
	free();
}

void NormalStimulator::propagate()
{
	if ( dst->evolve_locally() ) {
		for ( NeuronID i = 0 ; i < dst->get_post_size() ; ++i ) {
			float draw = (*die)();
			target_vector->data[i] = draw; 
		}
	}
}

void NormalStimulator::set_sigma(AurynWeight sigma) {
	delete dist;
	normal_sigma = sigma;
	dist = new boost::normal_distribution<float> (0.0, get_lambda());
}

AurynFloat NormalStimulator::get_sigma() {
	return (AurynFloat) normal_sigma;
}

AurynFloat NormalStimulator::get_lambda() {
	return get_sigma();
}

void NormalStimulator::set_target_state(std::string state_name) {
	target_vector = dst->get_state_vector(state_name);
}

void NormalStimulator::seed(int s)
{
		gen.seed(s); 
}

