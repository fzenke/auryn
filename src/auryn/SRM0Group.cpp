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

#include "SRM0Group.h"

using namespace auryn;

boost::mt19937 SRM0Group::gen = boost::mt19937(); 

SRM0Group::SRM0Group(NeuronID size) : NeuronGroup(size)
{
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void SRM0Group::calculate_scale_constants()
{
	scale_mem  = auryn_timestep/tau_mem;
	scale_syn  = std::exp(-auryn_timestep/tau_syn);
}

void SRM0Group::init()
{
	e_rest = -60e-3;
	e_rev = -80e-3;
	thr = -50e-3;
	tau_mem = 10e-3;
	tau_syn = 5e-3;

	rho0 = 100.0;
	delta_u = 1e-4;

	calculate_scale_constants();
	syn_current = get_state_vector("syn_current");
	warped_lifetime = get_state_vector("warped_time");
	temp = get_state_vector("_temp");

	default_exc_target_state = syn_current;
	default_inh_target_state = syn_current;

	dist = new boost::exponential_distribution<>(1.0);;
	die  = new boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > ( gen, *dist );
	salt = sys->get_seed();
	seed(sys->get_seed());

	clear();
}

void SRM0Group::draw(NeuronID i)
{
	const AurynDouble r = (*die)();
	warped_lifetime->set(i,r);
}

void SRM0Group::draw_all()
{
	for ( NeuronID i = 0 ; i < get_post_size() ; ++i ) {
		draw(i);
	}
}

void SRM0Group::clear()
{
	clear_spikes();
    mem->set_all( e_rest);
	draw_all();
}


SRM0Group::~SRM0Group()
{
	if ( !evolve_locally() ) return;

	delete dist;
	delete die;
}


void SRM0Group::evolve()
{
	// integrate membrane
    // compute current
    temp->diff(e_rest, mem); // leak current
	temp->add(syn_current); // syn_current

    // membrane dynamics
    mem->saxpy(scale_mem,temp);

	// lifetime decrement
	// compute instantaneous firing rate
	temp->diff(mem,thr);
	temp->mul(1.0/delta_u); 
	temp->fast_exp();
	// temp->set_all(1.0); // for testing at fixed voltage values
	temp->mul(auryn_timestep*rho0); 


	// decrease ttls by warped time
	warped_lifetime->sub(temp);

	// hard refractory time (clamped to zero)
	for (NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
		// stochastic spike generation TODO
		if (warped_lifetime->get(i)<0.0) {
			push_spike(i);
			draw(i);
			mem->set(i, e_rest);
		} 
	}

	syn_current->scale(scale_syn);
}

void SRM0Group::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}

void SRM0Group::seed(unsigned int s)
{
	std::stringstream oss;
	oss << "SRM0Group:: Seeding with " << s
		<< " and " << salt << " salt";
	auryn::logger->msg(oss.str(),NOTIFICATION);

	gen.seed( s + salt );  
}
