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

#include "IF2Group.h"


IF2Group::IF2Group( NeuronID size, AurynFloat load, NeuronID total ) : NeuronGroup(size,load,total)
{
	sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void IF2Group::calculate_scale_constants()
{
	scale_ampa =  exp(-dt/tau_ampa) ;
	scale_gaba =  exp(-dt/tau_gaba) ;
	scale_thr = exp(-dt/tau_thr) ;
}

void IF2Group::init()
{
	e_rest = -70e-3;
	e_rev = -80e-3;
	e_nmda_onset = -65e-3;
	nmda_slope = 1.0/60e-3;
	thr_rest = -50e-3;
	dthr = 100e-3;
	tau_thr = 5e-3;
	tau_mem = 20e-3;
	tau_ampa = 5e-3;
	tau_gaba = 10e-3;
	tau_nmda = 100e-3;

	set_ampa_nmda_ratio(1.0);

	calculate_scale_constants();
	
	// thr = auryn_vector_float_alloc (size); 
	
	t_leak = get_state_vector("t_leak");
	t_exc =  get_state_vector("t_exc");
	t_inh = get_state_vector("t_inh");
	nmda_opening = get_state_vector("nmda_opening");

	clear();
}

void IF2Group::clear()
{
	clear_spikes();
	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   auryn_vector_float_set (mem, i, e_rest);
	   auryn_vector_float_set (thr, i, 0.);
	   auryn_vector_float_set (g_ampa, i, 0.);
	   auryn_vector_float_set (g_gaba, i, 0.);
	   auryn_vector_float_set (g_nmda, i, 0.);
	}
}

void IF2Group::free() {
}

IF2Group::~IF2Group()
{
	if ( evolve_locally() ) free();
}

void IF2Group::integrate_nonlinear_nmda_synapses()
{
	// decay of ampa and gaba channel, i.e. multiply by exp(-dt/tau)
    auryn_vector_float_scale(scale_ampa,g_ampa);
    auryn_vector_float_scale(scale_gaba,g_gaba);

    // compute dg_nmda = (g_ampa-g_nmda)*dt/tau_nmda and add to g_nmda
	AurynFloat mul_nmda = dt/tau_nmda;
    auryn_vector_float_saxpy(mul_nmda,g_ampa,g_nmda);
	auryn_vector_float_saxpy(-mul_nmda,g_nmda,g_nmda);

	// BEGIN implement NMDA voltage dependence
	auryn_vector_float_copy( mem, nmda_opening);
	auryn_vector_float_add_constant( nmda_opening , -e_nmda_onset );
	auryn_vector_float_scale( nmda_slope, nmda_opening );
	for ( AurynState * ptr = auryn_vector_float_ptr( nmda_opening , 0 ) ; ptr != auryn_vector_float_ptr( nmda_opening , get_post_size()-1 )+1 ; ++ptr ) {
		AurynFloat x = *ptr;
		AurynFloat x2 = x*x;
		AurynFloat r = x2/(1.0+x2);
		if (x>0) *ptr = r; // rectification
		else *ptr = 0;
		// cout << *ptr << endl;
	}
	// END implement NMDA voltage dependence
	
    // excitatory
    auryn_vector_float_copy(g_nmda,t_exc);
    auryn_vector_float_scale(-A_nmda,t_exc);
	auryn_vector_float_mul(t_exc,nmda_opening);
    auryn_vector_float_saxpy(-A_ampa,g_ampa,t_exc);
    auryn_vector_float_mul(t_exc,mem);
    
    // inhibitory
    auryn_vector_float_copy(mem,t_inh);
    auryn_vector_float_add_constant(t_inh,-e_rev);
    auryn_vector_float_mul(t_inh,g_gaba);
}

/// Integrate the internal state
/*!
       This method applies the Euler integration step to the membrane dynamics.
 */
void IF2Group::integrate_membrane()
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

void IF2Group::check_thresholds()
{
	auryn_vector_float_clip( mem, e_rev );

	AurynState * thr_ptr = thr->data;
	for ( AurynState * i = mem->data ; i != mem->data+get_rank_size() ; ++i ) { // it's important to use rank_size here otherwise there might be spikes from units that do not exist
    	if ( *i > ( thr_rest + *thr_ptr ) ) {
			NeuronID unit = i-mem->data;
			push_spike(unit);
		    set_val (mem, unit, e_rest); // reset
	        set_val (thr, unit, dthr); //refractory
		} 
		thr_ptr++;
	}

}

void IF2Group::evolve()
{
	integrate_nonlinear_nmda_synapses();
	integrate_membrane();
	check_thresholds();
}


void IF2Group::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}

AurynFloat IF2Group::get_tau_mem()
{
	return tau_mem;
}

void IF2Group::set_tau_ampa(AurynFloat taum)
{
	tau_ampa = taum;
	calculate_scale_constants();
}

AurynFloat IF2Group::get_tau_ampa()
{
	return tau_ampa;
}

void IF2Group::set_tau_gaba(AurynFloat taum)
{
	tau_gaba = taum;
	calculate_scale_constants();
}

AurynFloat IF2Group::get_tau_gaba()
{
	return tau_gaba;
}

void IF2Group::set_tau_nmda(AurynFloat taum)
{
	tau_nmda = taum;
	calculate_scale_constants();
}

AurynFloat IF2Group::get_tau_nmda()
{
	return tau_nmda;
}

void IF2Group::set_ampa_nmda_ratio(AurynFloat ratio) 
{
 	A_ampa = ratio/(ratio+1.0);
	A_nmda = 1./(ratio+1.0);
}
