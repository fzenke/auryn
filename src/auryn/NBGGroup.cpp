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



#include "NBGGroup.h"

using namespace auryn;

NBGGroup::NBGGroup( NeuronID size, NodeDistributionMode distmode ) : NeuronGroup(size, distmode)
{
	sys->register_spiking_group(this);
	set_name("NBGGroup");
	if ( evolve_locally() ) init();
}

void NBGGroup::precalculate_constants()
{
	scale_ampa  =  std::exp(-auryn_timestep/tau_ampa);
	scale_gaba  =  std::exp(-auryn_timestep/tau_gaba);
	mul_nmda    = 1.0-std::exp(-auryn_timestep/tau_nmda);

	mul_thr     = auryn_timestep/tau_thr;
	mul_soma    = auryn_timestep*(gs/Cs);
	mul_wsoma   = auryn_timestep*(b_wsoma/Cs);
	mul_dend    = auryn_timestep*(gd/Cd);
	mul_wdend   = auryn_timestep*(a_wdend/Cd);

	mul_alpha = auryn_timestep*(alpha/Cs); 
	mul_beta = auryn_timestep*(beta/Cd);

	// mul_wsoma = std::exp(-auryn_timestep/tau_wsoma);

	scale_wsoma = std::exp(-auryn_timestep/tau_wsoma); 
	scale_wdend = std::exp(-auryn_timestep/tau_wdend); 
	aux_a_wdend = auryn_timestep*(1.0/tau_wdend);
}

void NBGGroup::init()
{
	e_rest = -70e-3;
	e_reset = e_rest;
	e_inh  = -80e-3;
	e_thr = -50e-3;

	// threshold
	tau_thr   = 5e-3;
	e_spk_thr = 1e-3; // TODO find good value here

	// soma 
	Cs = 370e-12;
	gs = 23e-9;

	// dendrite
	Cd = 170e-12;
    gd = 24e-9;


	alpha = 1300e-12; // dendritic coupling strength to soma
	beta  = 1200e-12; // somatic coupling strength to dendrite
	box_height = auryn_timestep*2600e-12/Cd;

	e_dend = -38e-3;
	const AurynFloat Dthresh = 6e-3;
	xi = 1.0/Dthresh;
	


	tau_wsoma = 100e-3;
	tau_wdend = 30e-3;
	b_wsoma = -200e-12; // adaptation current amplitude
	a_wdend = -13e-9; 

	// synaptic time constants
	tau_ampa = 5e-3;
	tau_gaba = 10e-3;
	tau_nmda = 100e-3;


	set_ampa_nmda_ratio(1.0);

	precalculate_constants();

	// declare state vectors
	state_soma = mem; // convention to use mem declared in NeuronGroup
	state_dend = get_state_vector("Vd");
	state_wsoma = get_state_vector("wsoma");
	state_wdend = get_state_vector("wdend");

	// counter to implement BAP box kernel
	post_spike_countdown = new AurynVector<unsigned int>(get_vector_size());
	const AurynFloat box_kernel_length = 2e-3;
	post_spike_reset = int(box_kernel_length/auryn_timestep);  

	// declare tmp vectos
	temp = get_state_vector("_temp");
	t_leak = get_state_vector("t_leak");
	t_exc =  get_state_vector("t_exc");
	t_inh = get_state_vector("t_inh");

	clear();
}

void NBGGroup::clear()
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

void NBGGroup::free() {
	delete post_spike_countdown;
}

NBGGroup::~NBGGroup()
{
	if ( evolve_locally() ) free();
}

void NBGGroup::integrate_linear_nmda_synapses()
{
    // excitatory
	t_exc->copy(g_ampa);
	t_exc->scale(-A_ampa);
	t_exc->saxpy(-A_nmda,g_nmda);
	t_exc->mul(mem);
    
    // inhibitory
	t_inh->diff(mem,e_inh);
	t_inh->mul(g_gaba);

    // compute dg_nmda = (g_ampa-g_nmda)*auryn_timestep/tau_nmda and add to g_nmda
	g_nmda->saxpy(mul_nmda, g_ampa);
	g_nmda->saxpy(-mul_nmda, g_nmda);

	// decay of ampa and gaba channel, i.e. multiply by exp(-auryn_timestep/tau)
	g_ampa->scale(scale_ampa);
	g_gaba->scale(scale_gaba);
}

/*! \brief This method applies the Euler integration step to the membrane dynamics. */
void NBGGroup::integrate_membrane()
{
    // somatic dynamics 
	temp->sigmoid(state_dend, xi, e_dend ); // sigmoid activation
	// temp->diff(state_dend, e_dend );
	// temp->mul(-xi);
	// temp->exp();
	// temp->add(1.0);
	// temp->inv();
	// temp->print();

	t_leak->diff(e_rest, state_soma); // leak current
    state_soma->saxpy(mul_soma, t_leak);
	state_soma->saxpy(mul_wsoma, state_wsoma);
	state_soma->saxpy(mul_alpha, temp); // nonlinear activation from dendrite

	state_wsoma->scale(scale_wsoma);
	
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
    state_dend->saxpy(mul_dend, t_exc);  // exc synaptic input in units of leak
    state_dend->saxpy(-mul_dend,t_inh); // inh synaptic input 

	state_wdend->scale(scale_wdend);
	temp->diff(state_dend,e_rest);  // dendritic leak
	state_wdend->saxpy(aux_a_wdend,temp);

	// decay moving threshold
	thr->follow_scalar(e_thr, mul_thr);

}

void NBGGroup::check_thresholds()
{
	// mem->clip( e_inh, 0.0 );

	AurynState * thr_ptr = thr->data;
	for ( AurynState * i = mem->data ; i != mem->data+get_rank_size() ; ++i ) { // it's important to use rank_size here otherwise there might be spikes from units that do not exist
    	if ( *i > *thr_ptr ) {
			NeuronID unit = i-mem->data;
			push_spike(unit);
		    mem->set( unit, e_reset); // reset
	        thr->set( unit, e_spk_thr); // increase dynamic threshold (refractory)
			state_wsoma->add_specific(unit,1.0); // increments somatic adaptation variable
			post_spike_countdown->set(unit,post_spike_reset);
		} 
		thr_ptr++;
	}

}

void NBGGroup::evolve()
{
	integrate_linear_nmda_synapses();
	integrate_membrane();
	check_thresholds();
}


void NBGGroup::set_tau_ampa(AurynFloat taum)
{
	tau_ampa = taum;
	precalculate_constants();
}

AurynFloat NBGGroup::get_tau_ampa()
{
	return tau_ampa;
}

void NBGGroup::set_tau_gaba(AurynFloat taum)
{
	tau_gaba = taum;
	precalculate_constants();
}

AurynFloat NBGGroup::get_tau_gaba()
{
	return tau_gaba;
}

void NBGGroup::set_tau_nmda(AurynFloat tau)
{
	if ( tau < tau_ampa ) { 
		logger->warning("tau_nmda has to be larger than tau_ampa in NBGGroup");
		return;
	}
	tau_nmda = tau;
	precalculate_constants();
}

void NBGGroup::set_tau_thr(AurynFloat tau)
{
	tau_thr = tau;
	precalculate_constants();
}


AurynFloat NBGGroup::get_tau_nmda()
{
	return tau_nmda;
}

void NBGGroup::set_ampa_nmda_ratio(AurynFloat ratio) 
{
 	A_ampa = ratio/(ratio+1.0);
	A_nmda = 1./(ratio+1.0);
}

void NBGGroup::set_nmda_ampa_current_ampl_ratio(AurynFloat ratio) 
{
	const double tau_r = tau_ampa;
	const double tau_d = tau_nmda;

	// compute amplitude of NMDA conductance
	const double tmax = tau_r*std::log((tau_r+tau_d)/tau_r); // argmax
	const double ampl = (1.0-std::exp(-tmax/tau_r))*std::exp(-tmax/tau_d)*tau_r/(tau_d-tau_r); 

	// set relative amplitudes
	A_ampa = 1.0;
	A_nmda = ratio/ampl;

	// normalize sum to one
	const double sum = A_ampa+A_nmda;
	A_ampa /= sum;
	A_nmda /= sum;

	// write constants to logfile
	logger->parameter("A_ampa", A_ampa);
	logger->parameter("A_nmda", A_nmda);
}
