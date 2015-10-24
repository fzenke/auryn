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

#include "PoissonGroup.h"

boost::mt19937 PoissonGroup::gen = boost::mt19937(); 

void PoissonGroup::init(AurynDouble  rate)
{
	sys->register_spiking_group(this);
	if ( evolve_locally() ) {

		dist = new boost::uniform_01<> ();
		die  = new boost::variate_generator<boost::mt19937&, boost::uniform_01<> > ( gen, *dist );
		seed(620349123); 
		x = 0;
		set_rate( rate );

		stringstream oss;
		oss << "PoissonGroup:: Seeding with " << communicator->rank();
		logger->msg(oss.str(),NOTIFICATION);
	}
}

PoissonGroup::PoissonGroup(NeuronID n, AurynDouble  rate ) : SpikingGroup( n , POISSON_LOAD_MULTIPLIER*rate ) 
{
	init(rate);
}

PoissonGroup::~PoissonGroup()
{
	if ( evolve_locally() ) {
		delete dist;
		delete die;
	}
}

void PoissonGroup::set_rate(AurynDouble  rate)
{
	lambda = 1.0/(1.0/rate-dt);
    if ( evolve_locally() ) {
		if ( rate > 0.0 ) {
		  AurynDouble r = -log((*die)()+1e-128)/lambda;
		  x = (NeuronID)(r/dt+0.5); 
		} else {
			// if the rate is zero this triggers one spike at the end of time/groupsize
			// this is the easiest way to take care of the zero rate case, which should 
			// be avoided in any case.
			x = std::numeric_limits<NeuronID>::max(); 
		}
    }
}

AurynDouble  PoissonGroup::get_rate()
{
	return lambda;
}


void PoissonGroup::evolve()
{
	while ( x < get_rank_size() ) {
		push_spike ( x );
		AurynDouble r = -log((*die)()+1e-128)/lambda;
		// we add 1.5: one to avoid two spikes per bin and 0.5 to 
		// compensate for rounding effects from casting
		x += (NeuronID)(r/dt+1.5); 
		// beware one induces systematic error that becomes substantial at high rates, but keeps neuron from spiking twice per time-step
	}
	x -= get_rank_size();
}

void PoissonGroup::seed(int s)
{
		gen.seed(s+communicator->rank()); // TODO solve seeding problem 
}

