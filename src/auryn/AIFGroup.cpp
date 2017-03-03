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

#include "AIFGroup.h"

using namespace auryn;


AIFGroup::AIFGroup( NeuronID size, NodeDistributionMode distmode) : NeuronGroup(size,distmode)
{
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void AIFGroup::calculate_scale_constants()
{
	scale_ampa =  exp(-auryn_timestep/tau_ampa) ;
	scale_gaba =  exp(-auryn_timestep/tau_gaba) ;
	scale_thr = exp(-auryn_timestep/tau_thr) ;
	scale_adapt1 = exp(-auryn_timestep/tau_adapt1);
}

void AIFGroup::init()
{
	e_rest = -70e-3;
	e_rev = -80e-3;
	thr_rest = -50e-3;
	dthr = 100e-3;
	tau_thr = 5e-3;
	tau_mem = 20e-3;
	tau_ampa = 5e-3;
	tau_gaba = 10e-3;
	tau_nmda = 100e-3;

	set_ampa_nmda_ratio(1.0);

	tau_adapt1 = 0.1;
	dg_adapt1  = 0.1;
 
	calculate_scale_constants();

	t_leak = get_state_vector("t_leak");
	t_exc  =  get_state_vector("t_exc");
	t_inh  = get_state_vector("t_inh");
	g_adapt1 = get_state_vector("g_adapt1");

	clear();
}

void AIFGroup::clear()
{
	clear_spikes();
	mem->set_all(e_rest);
	thr->set_all(0.);
	g_ampa->set_all(0.);
	g_gaba->set_all(0.);
	g_nmda->set_all(0.);
	g_adapt1->set_all(0.);

}


void AIFGroup::random_adapt(AurynState mean, AurynState sigma)
{
	boost::mt19937 ng_gen(42); // produces same series every time 
	boost::normal_distribution<> dist((double)mean, (double)sigma);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(ng_gen, dist);
	AurynState rv;

	for ( AurynLong i = 0 ; i<get_rank_size() ; ++i ) {
		rv = die();
		if ( rv>0 ) 
			g_adapt1->set( i, rv ); 
	}

	init_state();
}

void AIFGroup::free()
{
}

AIFGroup::~AIFGroup()
{
	if ( evolve_locally() ) free();
}

void AIFGroup::integrate_linear_nmda_synapses()
{
	// decay of ampa and gaba channel, i.e. multiply by exp(-auryn_timestep/tau)
	g_ampa->scale(scale_ampa);
	g_gaba->scale(scale_gaba);
	g_adapt1->scale(scale_adapt1);

    // compute dg_nmda = (g_ampa-g_nmda)*auryn_timestep/tau_nmda and add to g_nmda
	const AurynFloat mul_nmda = auryn_timestep/tau_nmda;
	g_nmda->saxpy(mul_nmda, g_ampa);
	g_nmda->saxpy(-mul_nmda, g_nmda);

    // excitatory
	t_exc->copy(g_ampa);
	t_exc->scale(-A_ampa);
	t_exc->saxpy(-A_nmda,g_nmda);
	t_exc->mul(mem);
    
    // inhibitory
    t_leak->copy(g_gaba); // using t_leak as temp here
    t_leak->saxpy(1.0, g_adapt1); 
	t_inh->diff(mem,e_rev);
	t_inh->mul(t_leak);
}

/// Integrate the internal state
/*!
       This method applies the Euler integration step to the membrane dynamics.
 */
void AIFGroup::integrate_membrane()
{
	// moving threshold
	thr->scale(scale_thr);
    
    // leak
	t_leak->diff(mem,e_rest);
    
    // membrane dynamics
	const AurynFloat mul_tau_mem = auryn_timestep/tau_mem;
    mem->saxpy(mul_tau_mem,t_exc);
    mem->saxpy(-mul_tau_mem,t_inh);
    mem->saxpy(-mul_tau_mem,t_leak);
}

void AIFGroup::check_thresholds()
{
	mem->clip( e_rev, 0.0 );

	AurynState * thr_ptr = thr->data;
	for ( AurynState * i = mem->data ; i != mem->data+get_rank_size() ; ++i ) { // it's important to use rank_size here otherwise there might be spikes from units that do not exist
    	if ( *i > ( thr_rest + *thr_ptr ) ) {
			NeuronID unit = i-mem->data;
			push_spike(unit);
		    mem->set( unit, e_rest); // reset
	        thr->set( unit, dthr); //refractory
			g_adapt1->add_specific( unit, dg_adapt1);
		} 
		thr_ptr++;
	}

}

void AIFGroup::evolve()
{
	integrate_linear_nmda_synapses();
	integrate_membrane();
	check_thresholds();
}


void AIFGroup::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}


void AIFGroup::set_tau_ampa(AurynFloat taum)
{
	tau_ampa = taum;
	calculate_scale_constants();
}

AurynFloat AIFGroup::get_tau_ampa()
{
	return tau_ampa;
}

void AIFGroup::set_tau_gaba(AurynFloat taum)
{
	tau_gaba = taum;
	calculate_scale_constants();
}

AurynFloat AIFGroup::get_tau_gaba()
{
	return tau_gaba;
}

void AIFGroup::set_tau_nmda(AurynFloat taum)
{
	tau_nmda = taum;
	calculate_scale_constants();
}

AurynFloat AIFGroup::get_tau_nmda()
{
	return tau_nmda;
}

void AIFGroup::set_tau_adapt(AurynFloat taua)
{
	tau_adapt1 = taua;
	calculate_scale_constants();
}

AurynFloat AIFGroup::get_tau_adapt()
{
	return tau_adapt1;
}


void AIFGroup::set_ampa_nmda_ratio(AurynFloat ratio) 
{
 	A_ampa = ratio/(ratio+1.0);
	A_nmda = 1./(ratio+1.0);
}
