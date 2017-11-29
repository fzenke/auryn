/* 
* Copyright 2014-2017 Friedemann Zenke
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



#include "NaudGroup.h"

using namespace auryn;

NaudGroup::NaudGroup( NeuronID size, NodeDistributionMode distmode ) : NeuronGroup(size, distmode)
{
	sys->register_spiking_group(this);
	set_name("NaudGroup");
	if ( evolve_locally() ) init();
}

void NaudGroup::precalculate_constants()
{
	scale_ampa = std::exp(-auryn_timestep/tau_ampa);
	scale_gaba = std::exp(-auryn_timestep/tau_gaba);
	mul_nmda   = 1.0-std::exp(-auryn_timestep/tau_nmda);


	mul_thr    = auryn_timestep/tau_thr;
	mul_soma   = auryn_timestep/tau_soma;
	mul_alpha  = auryn_timestep*(gs/Cs); 
	mul_wsoma  = auryn_timestep*(b_wsoma/Cs);

	mul_dend   = auryn_timestep/tau_dend; // leak
	mul_beta   = auryn_timestep*(gd/Cd);  // dend. nonlinearity 
	box_height = auryn_timestep*(zeta/Cd); // or bAP
	mul_wdend  = auryn_timestep*(a_wdend/Cd);


	scale_wsoma = std::exp(-auryn_timestep/tau_wsoma); 
	aux_mul_wdend = auryn_timestep/tau_wdend; // FIXME
}

void NaudGroup::init()
{
	e_rest = -70e-3;
	e_reset = e_rest;
	e_inh  = -80e-3;
	e_thr = -50e-3;

	// threshold
	tau_thr   = 27e-3;
	e_spk_thr = 2e-3; 

	// soma 
	tau_soma = 16e-3;
	Cs = 370e-12;
	gs = 1300e-12; // somatic coupling strength to dendrite
	tau_wsoma = 100e-3;
	b_wsoma = -200e-12; // adaptation current amplitude

	// dendrite
	tau_dend = 7e-3;
	Cd = 170e-12;
    gd = 1200e-12; // dendritic coupling strength to soma
	tau_wdend = 30e-3;
	a_wdend = -13e-9; 

	// alpha = 1300e-12; 
	// beta  = 1200e-12; 
	zeta  = 2600e-12; // bap strength
	box_kernel_length = 2e-3; // in ms

	e_dend = -38e-3;
	const AurynFloat Dthresh = 6e-3;
	xi = 1.0/Dthresh;
	

	// synaptic time constants
	tau_ampa = 5e-3;
	tau_gaba = 10e-3;
	tau_nmda = 100e-3;


	precalculate_constants();

	// declare state vectors
	state_soma = mem; // convention to use mem declared in NeuronGroup
	state_dend = get_state_vector("Vd");
	state_wsoma = get_state_vector("wsoma");
	state_wdend = get_state_vector("wdend");

	// counter to implement BAP box kernel
	post_spike_countdown = new AurynVector<unsigned int>(get_vector_size());
	post_spike_reset = int(box_kernel_length/auryn_timestep);  

	// declare tmp vectos
	temp = get_state_vector("_temp");
	t_leak = get_state_vector("t_leak");
	t_exc =  get_state_vector("t_exc");
	t_inh = get_state_vector("t_inh");


	syn_current_exc_soma = get_state_vector("syn_current_exc_soma");
	syn_current_exc_dend = get_state_vector("syn_current_exc_dend");
	syn_current_inh_soma = get_state_vector("syn_current_inh_soma");
	syn_current_inh_dend = get_state_vector("syn_current_inh_dend");

	// set up synaptic input
	g_ampa_dend = get_state_vector("g_ampa_dend");
	g_gaba_dend = get_state_vector("g_gaba_dend");

	syn_exc_soma = new LinearComboSynapse(this, g_ampa, syn_current_exc_soma );
	syn_exc_soma->set_ampa_nmda_ratio(1.0);
	syn_exc_soma->set_tau_ampa(tau_ampa);
	syn_exc_soma->set_tau_nmda(tau_nmda);
	syn_exc_soma->set_e_rev(0.0);

	syn_exc_dend = new LinearComboSynapse(this, g_ampa_dend, syn_current_exc_dend );
	syn_exc_dend->set_ampa_nmda_ratio(1.0);
	syn_exc_dend->set_tau_ampa(tau_ampa);
	syn_exc_dend->set_tau_nmda(tau_nmda);
	syn_exc_dend->set_e_rev(0.0);

	syn_inh_soma = new ExpCobaSynapse(this, g_gaba, syn_current_inh_soma );
	syn_inh_soma->set_tau(tau_gaba);
	syn_inh_soma->set_e_rev(e_inh);

	syn_inh_dend = new ExpCobaSynapse(this, g_gaba_dend, syn_current_inh_dend );
	syn_inh_dend->set_tau(tau_gaba);
	syn_inh_dend->set_e_rev(e_inh);

	clear();
}

void NaudGroup::clear()
{
	clear_spikes();
	thr->set_all(e_thr);
	mem->set_all(e_rest);
	state_dend->set_all(e_rest);
	state_wsoma->set_all(0.0);
	state_wdend->set_all(1e-4);

	// zero synaptic input
	g_ampa->set_zero();
	g_gaba->set_zero();
	g_nmda->set_zero();
}

void NaudGroup::free() {
	delete syn_exc_dend;
	delete syn_inh_dend;
	delete syn_exc_soma;
	delete syn_inh_soma;
	delete post_spike_countdown;
}

NaudGroup::~NaudGroup()
{
	if ( evolve_locally() ) free();
}

/*! \brief This method applies the Euler integration step to the membrane dynamics. */
void NaudGroup::integrate_membrane()
{
    // somatic dynamics 
	temp->sigmoid(state_dend, xi, e_dend ); // sigmoid activation
	t_leak->diff(e_rest, state_soma); // leak current
    state_soma->saxpy(mul_soma, t_leak);
	state_soma->saxpy(mul_wsoma, state_wsoma);
	state_soma->saxpy(mul_alpha, temp); // nonlinear activation from dendrite
	
	// dendritic dynamics
	t_leak->diff(e_rest, state_dend);  // dendritic leak
    state_dend->saxpy(mul_dend, t_leak); 
	state_dend->saxpy(mul_wdend, state_wdend);
	state_dend->saxpy(mul_beta, temp); // nonlinear activation from dendrite

	// box kernel for BAP
	for ( NeuronID i = 0 ; i < get_post_size() ; ++i ) {
		if ( post_spike_countdown->get(i) ) {
			state_dend->add_specific(i, box_height); 

			// decrease post spike counter
			post_spike_countdown->add_specific(i,-1);
		} }

	// synaptic input 
    state_soma->saxpy(mul_soma, syn_current_exc_soma); 
    state_soma->saxpy(mul_soma, syn_current_inh_soma); 
    state_dend->saxpy(mul_dend, syn_current_exc_dend); 
    state_dend->saxpy(mul_dend, syn_current_inh_dend); 

	// update somatic adapation variable (S42)
	state_wsoma->scale(scale_wsoma);

	// update dendritic adaptation variable (S44)
	temp->diff(state_dend,e_rest);  // dendritic leak
	temp->saxpy(-1.0,state_wdend);
	state_wdend->saxpy(aux_mul_wdend,temp); 

	// decay moving threshold
	thr->follow_scalar(e_thr, mul_thr);
}

void NaudGroup::check_thresholds()
{
	// mem->clip( e_inh, 0.0 );

	AurynState * thr_ptr = thr->data;
	for ( AurynState * i = mem->data ; i != mem->data+get_rank_size() ; ++i ) { // it's important to use rank_size here otherwise there might be spikes from units that do not exist
    	if ( *i > *thr_ptr ) {
			NeuronID unit = i-mem->data;
			push_spike(unit);
		    mem->set( unit, e_reset); // reset
	        thr->add_specific( unit, e_spk_thr); // increase dynamic threshold (refractory)
			state_wsoma->add_specific(unit, 1.0); // increments somatic adaptation variable
			post_spike_countdown->set(unit, post_spike_reset);
		} 
		thr_ptr++;
	}

}

void NaudGroup::evolve()
{
	syn_exc_soma->evolve(); //!< integrate_linear_nmda_synapses
	syn_inh_soma->evolve(); 
	syn_exc_dend->evolve(); //!< integrate_linear_nmda_synapses
	syn_inh_dend->evolve(); 
	// integrate_linear_nmda_synapses();
	integrate_membrane();
	check_thresholds();
}


void NaudGroup::set_tau_thr(AurynFloat tau)
{
	tau_thr = tau;
	precalculate_constants();
}

