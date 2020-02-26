/* 
* Copyright 2014-2020 Friedemann Zenke
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

#include "ModSynIFGroup.h"

using namespace auryn;

ModSynIFGroup::ModSynIFGroup( NeuronID size, NodeDistributionMode distmode ) : NeuronGroup(size, distmode)
{
	sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void ModSynIFGroup::calculate_scale_constants()
{
	scale_thr  = std::exp(-auryn_timestep/tau_thr) ;
}

void ModSynIFGroup::init()
{
	e_rest = -70e-3;
	e_reset = -70e-3;
	e_rev = -80e-3;
	thr_rest = -50e-3;
	dthr = 100e-3;
	tau_thr = 5e-3;
	tau_mem = 20e-3;
	tau_ampa = 5e-3;
	tau_gaba = 10e-3;
	tau_nmda = 100e-3;

	t_leak = get_state_vector("t_leak");
	syn_current_exc =  get_state_vector("syn_current_exc");
	syn_current_inh = get_state_vector("syn_current_inh");

	exc_synapses = new LinearComboSynapse(this, g_ampa, syn_current_exc );
	exc_synapses->set_ampa_nmda_ratio(1.0);
	exc_synapses->set_tau_ampa(tau_ampa);
	exc_synapses->set_tau_nmda(tau_nmda);
	exc_synapses->set_e_rev(0.0);

	inh_synapses = new ExpCobaSynapse(this, g_gaba, syn_current_inh );
	inh_synapses->set_tau(tau_gaba);
	inh_synapses->set_e_rev(e_rev);

	calculate_scale_constants();
	
	clear();
}

void ModSynIFGroup::clear()
{
	clear_spikes();
	mem->set_all(e_rest);
	thr->set_zero();
	g_ampa->set_zero();
	g_gaba->set_zero();
	g_nmda->set_zero();
}

void ModSynIFGroup::free() {
	delete exc_synapses;
	delete inh_synapses;
}

ModSynIFGroup::~ModSynIFGroup()
{
	if ( evolve_locally() ) free();
}

/// Integrate the internal state
/*!
       This method applies the Euler integration step to the membrane dynamics.
 */
void ModSynIFGroup::integrate_membrane()
{
	// moving threshold
	thr->scale(scale_thr);
    
    // leak
	t_leak->diff(e_rest,mem);
    
    // membrane dynamics
	const AurynFloat mul_tau_mem = auryn_timestep/tau_mem;
    mem->saxpy(mul_tau_mem,syn_current_exc); // syn_current_exc is computed by combo synapse object
    mem->saxpy(mul_tau_mem,syn_current_inh);
    mem->saxpy(mul_tau_mem,t_leak);
}

void ModSynIFGroup::check_thresholds()
{
	mem->clip( e_rev, 0.0 );

	AurynState * thr_ptr = thr->data;
	for ( AurynState * i = mem->data ; i != mem->data+get_rank_size() ; ++i ) { // it's important to use rank_size here otherwise there might be spikes from units that do not exist
    	if ( *i > ( thr_rest + *thr_ptr ) ) {
			NeuronID unit = i-mem->data;
			push_spike(unit);
		    mem->set( unit, e_reset); // reset
	        thr->set( unit, dthr); //refractory
		} 
		thr_ptr++;
	}

}

void ModSynIFGroup::evolve()
{
	exc_synapses->evolve(); //!< integrate_linear_nmda_synapses
	inh_synapses->evolve(); 
	integrate_membrane();
	check_thresholds();
}


void ModSynIFGroup::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}

AurynFloat ModSynIFGroup::get_tau_mem()
{
	return tau_mem;
}

void ModSynIFGroup::set_tau_ampa(AurynFloat taum)
{
	tau_ampa = taum;
	calculate_scale_constants();
}

AurynFloat ModSynIFGroup::get_tau_ampa()
{
	return tau_ampa;
}

void ModSynIFGroup::set_tau_gaba(AurynFloat taum)
{
	tau_gaba = taum;
	calculate_scale_constants();
}

AurynFloat ModSynIFGroup::get_tau_gaba()
{
	return tau_gaba;
}

void ModSynIFGroup::set_tau_nmda(AurynFloat tau)
{
	if ( tau < tau_ampa ) { 
		logger->warning("tau_nmda has to be larger than tau_ampa in ModSynIFGroup");
		return;
	}
	tau_nmda = tau;
	calculate_scale_constants();
}

void ModSynIFGroup::set_tau_thr(AurynFloat tau)
{
	tau_thr = tau;
	calculate_scale_constants();
}


AurynFloat ModSynIFGroup::get_tau_nmda()
{
	return tau_nmda;
}

void ModSynIFGroup::set_nmda_ampa_current_ampl_ratio(AurynFloat ratio)
{
	exc_synapses->set_nmda_ampa_current_ampl_ratio(ratio);
}

void ModSynIFGroup::set_ampa_nmda_ratio(AurynFloat ratio)
{
	exc_synapses->set_ampa_nmda_ratio(ratio);
}
