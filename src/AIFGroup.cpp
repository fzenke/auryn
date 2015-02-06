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

#include "AIFGroup.h"


AIFGroup::AIFGroup( NeuronID size, AurynFloat load, NeuronID total ) : NeuronGroup(size,load,total)
{
	sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void AIFGroup::calculate_scale_constants()
{
	scale_ampa =  exp(-dt/tau_ampa) ;
	scale_gaba =  exp(-dt/tau_gaba) ;
	scale_thr = exp(-dt/tau_thr) ;
	scale_adapt1 = exp(-dt/tau_adapt1);
}

void AIFGroup::init()
{
	e_rest = -70e-3;
	e_rev = -80e-3;
	thr_rest = -50e-3;
	dthr = 100e-3;
	tau_thr = 2e-3;
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
	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   auryn_vector_float_set (mem, i, e_rest);
	   auryn_vector_float_set (thr, i, 0.);
	   auryn_vector_float_set (g_ampa, i, 0.);
	   auryn_vector_float_set (g_gaba, i, 0.);
	   auryn_vector_float_set (g_nmda, i, 0.);
	   auryn_vector_float_set (g_adapt1, i, 0.);
	 }

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
			set_val (g_adapt1, i, rv ); 
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
	// decay of ampa and gaba channel, i.e. multiply by exp(-dt/tau)
    auryn_vector_float_scale(scale_ampa,g_ampa);
    auryn_vector_float_scale(scale_gaba,g_gaba);
    auryn_vector_float_scale(scale_adapt1,g_adapt1);

    // compute dg_nmda = (g_ampa-g_nmda)*dt/tau_nmda and add to g_nmda
	AurynFloat mul_nmda = dt/tau_nmda;
    auryn_vector_float_saxpy(mul_nmda,g_ampa,g_nmda);
	auryn_vector_float_saxpy(-mul_nmda,g_nmda,g_nmda);

    // excitatory
    auryn_vector_float_copy(g_ampa,t_exc);
    auryn_vector_float_scale(-A_ampa,t_exc);
    auryn_vector_float_saxpy(-A_nmda,g_nmda,t_exc);
    auryn_vector_float_mul(t_exc,mem);
    
    // inhibitory
    auryn_vector_float_copy(g_gaba,t_leak); // using t_leak as temp here
    auryn_vector_float_saxpy(1,g_adapt1,t_leak);
    auryn_vector_float_copy(mem,t_inh);
    auryn_vector_float_add_constant(t_inh,-e_rev);
    auryn_vector_float_mul(t_inh,t_leak);
}

/// Integrate the internal state
/*!
       This method applies the Euler integration step to the membrane dynamics.
 */
void AIFGroup::integrate_membrane()
{
	// moving threshold
    auryn_vector_float_scale(scale_thr,thr);
    
    // leak
	auryn_vector_float_copy(mem,t_leak);
    auryn_vector_float_add_constant(t_leak,-e_rest);
    
    // membrane dynamics
	AurynFloat mul_tau_mem = dt/tau_mem;
    auryn_vector_float_saxpy(mul_tau_mem,t_exc,mem);
    auryn_vector_float_saxpy(-mul_tau_mem,t_inh,mem);
    auryn_vector_float_saxpy(-mul_tau_mem,t_leak,mem);
}

void AIFGroup::check_thresholds()
{
	auryn_vector_float_clip( mem, e_rev );

	AurynState * thr_ptr = thr->data;
	for ( AurynState * i = mem->data ; i != mem->data+get_rank_size() ; ++i ) { // it's important to use rank_size here otherwise there might be spikes from units that do not exist
    	if ( *i > ( thr_rest + *thr_ptr ) ) {
			NeuronID unit = i-mem->data;
			push_spike(unit);
		    set_val (mem, unit, e_rest); // reset
	        set_val (thr, unit, dthr); //refractory
			add_val (g_adapt1, unit, dg_adapt1);
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
