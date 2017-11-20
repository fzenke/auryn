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
	scale_ampa  =  std::exp(-auryn_timestep/tau_ampa) ;
	scale_gaba  =  std::exp(-auryn_timestep/tau_gaba) ;
	scale_thr   = std::exp(-auryn_timestep/tau_thr) ;
	mul_nmda    = 1.0-std::exp(-auryn_timestep/tau_nmda);

	mul_soma    = auryn_timestep*(gs/Cs);
	mul_dend    = auryn_timestep*(gd/Cd);

	mul_ws = std::exp(-auryn_timestep/tau_ws);
	mul_wd = std::exp(-auryn_timestep/tau_wd);

	mul_alpha = auryn_timestep*(alpha/Cs);
	mul_beta = auryn_timestep*(beta/Cs);

	mul_adapt = auryn_timestep*(jump_w/Cs);
}

void NBGGroup::init()
{
	e_rest = -70e-3;
	e_reset = e_rest;
	e_inh  = -80e-3;
	e_thr = -50e-3;

	// threshold
	tau_thr  = 27e-3;

	// soma 
	Cs = 370e-12;
	gs = 23e-9;

	// dendrite
	Cd = 170e-12;
    gd = 24e-9;

	tau_ws = 100e-3;
	tau_wd = 30e-3;

	alpha = 567e-12;
	beta  = -207e-12;
	gamma = -207e-12;

	e_dend = -38e-3;
	Dt = 6e-3;

	jump_w = -200e-12; // adaptation current amplitude

	// synaptic time constants
	tau_ampa = 5e-3;
	tau_gaba = 10e-3;
	tau_nmda = 100e-3;

	tau_w = 100e-3;


	set_ampa_nmda_ratio(1.0);

	precalculate_constants();

	// declare state vectors
	state_soma = mem; // convention to use mem declared in NeuronGroup
	state_dend = get_state_vector("Vd");
	state_m  = get_state_vector("m");
	state_x  = get_state_vector("x");
	state_Vt = thr; // convention to use thr declared in NeuronGroup

	// mods (this deviates from the original kernel-based model
	state_w  = get_post_trace(tau_w);

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
	mem->set_all(e_rest);
	thr->set_all(e_thr);
	state_dend->set_all(e_rest);

	// zero synaptic input
	g_ampa->set_zero();
	g_gaba->set_zero();
	g_nmda->set_zero();
}

void NBGGroup::free() {
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
	// decay moving threshold
	state_Vt->follow_scalar(e_thr, scale_thr);
    
    // somatic dynamics 
	temp->sigmoid(state_dend, e_dend, 1.0f/Dt); // sigmoid activation

	t_leak->diff(e_rest, state_soma); // leak current
    state_soma->saxpy(mul_soma,t_leak);
	state_soma->saxpy(mul_alpha, state_m);
	state_soma->saxpy(mul_adapt, state_w);
	state_soma->saxpy(gs, state_w);
	// TODO add epsilon filtered current
	
	// dendritic dynamics
	t_leak->diff(e_rest, state_dend); 
    state_dend->saxpy(mul_dend,t_leak); // leak current
    state_dend->saxpy(mul_dend,t_exc);  // exc synaptic input
    state_dend->saxpy(-mul_dend,t_inh); // inh synaptic input
	state_dend->saxpy(mul_g1,state_m);
	state_dend->saxpy(mul_g2,state_x);
	// TODO add BAP
	// TODO add epsilonsd filtered somatic current
}

/*! Integrates equations (3) and (4) of model description */
void NBGGroup::integrate_active_dendritic_currents()
{
	// compute sigmoidal nonlinearity for Ca current (m-dynamics)
	// temp->diff(e_dend,state_dend);
	// temp->div(Dt);
	// temp->exp();
	// temp->add(1.0);
	// temp->inv();
	state_m->follow(temp, mul_m);

	// compute update for dx/dt
	state_x->follow(state_m, mul_x);
}

void NBGGroup::check_thresholds()
{
	mem->clip( e_inh, 0.0 );

	AurynState * thr_ptr = thr->data;
	for ( AurynState * i = mem->data ; i != mem->data+get_rank_size() ; ++i ) { // it's important to use rank_size here otherwise there might be spikes from units that do not exist
    	if ( *i > *thr_ptr ) {
			NeuronID unit = i-mem->data;
			push_spike(unit);
		    mem->set( unit, e_reset); // reset
	        thr->set( unit, Dt); // increase dynamic threshold (refractory)

		} 
		thr_ptr++;
	}

}

void NBGGroup::evolve()
{
	integrate_linear_nmda_synapses();
	integrate_membrane();
	integrate_active_dendritic_currents();
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
