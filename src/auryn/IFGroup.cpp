/* 
* Copyright 2014-2025 Friedemann Zenke
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

#include "IFGroup.h"

using namespace auryn;

IFGroup::IFGroup( NeuronID size, NodeDistributionMode distmode ) : NeuronGroup(size, distmode)
{
	sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void IFGroup::calculate_scale_constants()
{
	scale_ampa =  std::exp(-auryn_timestep/tau_ampa) ;
	scale_gaba =  std::exp(-auryn_timestep/tau_gaba) ;
	scale_thr  = std::exp(-auryn_timestep/tau_thr) ;
	mul_nmda   = 1.0-std::exp(-auryn_timestep/tau_nmda);
}

void IFGroup::init()
{
	u_rest = -70e-3;
	u_reset = -70e-3;
	u_inh_rev = -80e-3;
	thr_rest = -50e-3;
	dthr = 100e-3;
	tau_thr = 5e-3;
	tau_mem = 20e-3;
	tau_ampa = 5e-3;
	tau_gaba = 10e-3;
	tau_nmda = 100e-3;

	set_ampa_nmda_ratio(1.0);

	calculate_scale_constants();
	
	t_leak = get_state_vector("t_leak");
	t_exc =  get_state_vector("t_exc");
	t_inh = get_state_vector("t_inh");

	default_exc_target_state = g_ampa;
	default_inh_target_state = g_gaba;

	clear();
}

void IFGroup::clear()
{
	clear_spikes();
	mem->set_all(u_rest);
	thr->set_zero();
	g_ampa->set_zero();
	g_gaba->set_zero();
	g_nmda->set_zero();
}

void IFGroup::free() {
}

IFGroup::~IFGroup()
{
	if ( evolve_locally() ) free();
}

void IFGroup::integrate_linear_nmda_synapses()
{
    // excitatory
	t_exc->copy(g_ampa);
	t_exc->scale(-A_ampa);
	t_exc->saxpy(-A_nmda,g_nmda);
	t_exc->mul(mem);
    
    // inhibitory
	t_inh->diff(mem,u_inh_rev);
	t_inh->mul(g_gaba);

    // compute dg_nmda = (g_ampa-g_nmda)*auryn_timestep/tau_nmda and add to g_nmda
	g_nmda->saxpy(mul_nmda, g_ampa);
	g_nmda->saxpy(-mul_nmda, g_nmda);

	// decay of ampa and gaba channel, i.e. multiply by exp(-auryn_timestep/tau)
	g_ampa->scale(scale_ampa);
	g_gaba->scale(scale_gaba);
}

/// Integrate the internal state
/*!
       This method applies the Euler integration step to the membrane dynamics.
 */
void IFGroup::integrate_membrane()
{
	// moving threshold
	thr->scale(scale_thr);
    
    // leak
	t_leak->diff(mem,u_rest);
    
    // membrane dynamics
	const AurynFloat mul_tau_mem = auryn_timestep/tau_mem;
    mem->saxpy(mul_tau_mem,t_exc);
    mem->saxpy(-mul_tau_mem,t_inh);
    mem->saxpy(-mul_tau_mem,t_leak);
}

void IFGroup::check_thresholds()
{
	mem->clip( u_inh_rev, 0.0 );

	AurynState * thr_ptr = thr->data;
	for ( AurynState * i = mem->data ; i != mem->data+get_rank_size() ; ++i ) { // it's important to use rank_size here otherwise there might be spikes from units that do not exist
    	if ( *i > ( thr_rest + *thr_ptr ) ) {
			NeuronID unit = i-mem->data;
			push_spike(unit);
		    mem->set( unit, u_reset); // reset
	        thr->set( unit, dthr); //refractory
		} 
		thr_ptr++;
	}

}

void IFGroup::evolve()
{
	integrate_linear_nmda_synapses();
	integrate_membrane();
	check_thresholds();
}


void IFGroup::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}

AurynFloat IFGroup::get_tau_mem()
{
	return tau_mem;
}

void IFGroup::set_tau_ampa(AurynFloat taum)
{
	tau_ampa = taum;
	calculate_scale_constants();
}

AurynFloat IFGroup::get_tau_ampa()
{
	return tau_ampa;
}

void IFGroup::set_tau_gaba(AurynFloat taum)
{
	tau_gaba = taum;
	calculate_scale_constants();
}

AurynFloat IFGroup::get_tau_gaba()
{
	return tau_gaba;
}

void IFGroup::set_tau_nmda(AurynFloat tau)
{
	if ( tau < tau_ampa ) { 
		logger->warning("tau_nmda has to be larger than tau_ampa in IFGroup");
		return;
	}
	tau_nmda = tau;
	calculate_scale_constants();
}

void IFGroup::set_tau_thr(AurynFloat tau)
{
	tau_thr = tau;
	calculate_scale_constants();
}


AurynFloat IFGroup::get_tau_nmda()
{
	return tau_nmda;
}

void IFGroup::set_ampa_nmda_ratio(AurynFloat ratio) 
{
 	A_ampa = ratio/(ratio+1.0);
	A_nmda = 1./(ratio+1.0);
}

void IFGroup::set_nmda_ampa_current_ampl_ratio(AurynFloat ratio) 
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
