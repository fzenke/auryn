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

#include "ProfilePoissonGroup.h"

using namespace auryn;

boost::mt19937 ProfilePoissonGroup::gen = boost::mt19937(); 

void ProfilePoissonGroup::init(AurynDouble  rate)
{
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) {
		lambda = rate;

		dist = new boost::uniform_01<> ();
		die  = new boost::variate_generator<boost::mt19937&, boost::uniform_01<> > ( gen, *dist );
		seed(sys->get_seed()); // seeding problem
		x = 0;
		jumpsize = 0;

		// creates flat profile 
		profile = get_state_vector("firing_rate_profile");
		set_flat_profile();

		std::stringstream oss;
		oss << "ProfilePoissonGroup:: Seeding with " << sys->mpi_rank();
		auryn::logger->msg(oss.str(),NOTIFICATION);
	}
}

ProfilePoissonGroup::ProfilePoissonGroup(NeuronID n, AurynDouble  rate ) : SpikingGroup( n ) 
{
	init(rate);
}

ProfilePoissonGroup::~ProfilePoissonGroup()
{
	if ( evolve_locally() ) {
		delete dist;
		delete die;
	}
}

void ProfilePoissonGroup::set_rate(AurynDouble  rate)
{
	lambda = rate;
    if ( evolve_locally() ) {
		if ( rate > 0.0 ) {
		  AurynDouble r = -log((*die)())/lambda;
		  jumpsize = (r/auryn_timestep);
		  x = 0;
		} else {
			// if the rate is zero this triggers one spike at the end of time/groupsize
			// this is the easiest way to take care of the zero rate case, which should 
			// be avoided in any case.
			x = std::numeric_limits<NeuronID>::max(); 
		}
    }
}

void ProfilePoissonGroup::normalize_profile()
{
	AurynDouble sum = 0.0;
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
		sum += profile->data[i];
	}

	AurynDouble normalization_factor = get_rank_size()/sum;
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
		profile->data[i] *= normalization_factor ;
	}
}

void ProfilePoissonGroup::set_flat_profile()
{
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
		profile->data[i] = 1.0;
	}
}

void ProfilePoissonGroup::set_profile( AurynFloat * newprofile ) 
{
	for ( NeuronID i = 0 ; i < get_size() ; ++i ) {
		if ( localrank( i ) ) 
			profile->data[global2rank(i)] = newprofile[i];
	}

	normalize_profile();

	std::stringstream oss;
	oss << "ProfilePoissonGroup:: Successfully set external profile." ;
	auryn::logger->msg(oss.str(),NOTIFICATION);
}

void ProfilePoissonGroup::set_profile( auryn_vector_float * newprofile ) 
{
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
		profile->data[i] = newprofile->data[i];
	}

	// normalize_profile();

	std::stringstream oss;
	oss << "ProfilePoissonGroup:: Successfully set external profile." ;
	auryn::logger->msg(oss.str(),NOTIFICATION);
}


void ProfilePoissonGroup::set_gaussian_profile(AurynDouble  mean, AurynDouble sigma, AurynDouble floor)
{
	for ( NeuronID i = 0 ; i < get_size() ; ++i ) {
		if ( localrank(i) )
			profile->data[global2rank(i)] = exp(-pow((i-mean),2)/(2*sigma*sigma))*(1.0-floor)+floor;
	}

	normalize_profile();

	std::stringstream oss;
	oss << "ProfilePoissonGroup:: Setting gaussian profile with mean=" 
		<< mean
		<< " and sigma="
		<< sigma
		<< " and floor="
		<< floor;
	auryn::logger->msg(oss.str(),VERBOSE);
}

AurynDouble  ProfilePoissonGroup::get_rate()
{
	return lambda;
}


void ProfilePoissonGroup::evolve()
{
	while ( x < get_rank_size() ) {
		// std::cout << x << std::endl;
		jumpsize -= profile->data[x];
		if ( jumpsize < 0 ) { // reached jump target -> spike
			push_spike ( x );
			AurynDouble r = -log((*die)()+1e-20)/lambda;
			jumpsize = r/auryn_timestep; 
		}
		x++;
	}
	x -= get_rank_size();
}

void ProfilePoissonGroup::seed(int s)
{
		gen.seed(s); 
}

