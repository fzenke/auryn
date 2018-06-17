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

#include "PoissonStimulator.h"

using namespace auryn;

boost::mt19937 PoissonStimulator::gen = boost::mt19937();

PoissonStimulator::PoissonStimulator(NeuronGroup * target, AurynFloat rate, AurynWeight w ) : Device( )
{
	init(target,rate,w);
}


void PoissonStimulator::init( NeuronGroup * target, AurynFloat rate, AurynWeight w )
{
	auryn::sys->register_device(this);
	dst = target;

	set_target_state();

	poisson_weight = w;
	poisson_rate = rate;


	std::stringstream oss;
	oss << std::scientific << "PoissonStimulator:: initializing with mean " << get_lambda();
	auryn::logger->msg(oss.str(),NOTIFICATION);

	seed(61093*sys->mpi_rank());
	dist = new boost::poisson_distribution<int> (get_lambda());
	die  = new boost::variate_generator<boost::mt19937&, boost::poisson_distribution<int> > ( gen, *dist );
}

void PoissonStimulator::free( ) 
{
	delete dist;
	delete die;
}


PoissonStimulator::~PoissonStimulator()
{
	free();
}

void PoissonStimulator::execute()
{
	if ( dst->evolve_locally() ) {
		for ( NeuronID i = 0 ; i < dst->get_post_size() ; ++i ) {
			int draw = (*die)();
			target_vector->data[i] += draw*poisson_weight; 
		}
	}
}

void PoissonStimulator::set_rate(AurynFloat rate) {
	delete dist;
	poisson_rate = rate;
	dist = new boost::poisson_distribution<int> (get_lambda());
}

AurynFloat PoissonStimulator::get_rate() {
	return poisson_rate;
}

AurynFloat PoissonStimulator::get_lambda() {
	return get_rate()*auryn_timestep;
}

void PoissonStimulator::set_target_state(std::string state_name) {
	target_vector = dst->get_state_vector(state_name);
}

void PoissonStimulator::seed(int s)
{
		gen.seed(s); 
}

