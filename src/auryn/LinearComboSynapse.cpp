/* 
* Copyright 2014-2023 Friedemann Zenke
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

#include "LinearComboSynapse.h"

using namespace auryn;

LinearComboSynapse::LinearComboSynapse(NeuronGroup * parent, AurynStateVector * input, AurynStateVector * output) 
	: SynapseModel( parent, input, output )
{
	if ( input == output ) {
		logger->warning("LinearComboSynapse requires input and output state to be different!");
	}

	set_tau_ampa(5e-3); // sets a default timescale
	set_tau_nmda(100e-3); // sets a default timescale
	set_e_rev(0.0); // sets default reversal potential

	g_ampa = input_state;
	g_nmda = parent_group->get_state_vector("g_nmda");
	temp   = parent_group->get_state_vector("_temp");

	set_ampa_nmda_ratio(1.0);
}

LinearComboSynapse::~LinearComboSynapse() 
{
}

void LinearComboSynapse::set_tau_ampa(const AurynState tau)
{
	tau_ampa = tau;
	scale_ampa = std::exp(-auryn_timestep/tau_ampa);
}

void LinearComboSynapse::set_tau_nmda(const AurynState tau)
{
	tau_nmda = tau;
	mul_nmda   = 1.0-std::exp(-auryn_timestep/tau_nmda);
}

void LinearComboSynapse::set_e_rev(const AurynState reversal_pot)
{
	e_rev = reversal_pot;
}

void LinearComboSynapse::evolve()
{
	temp->diff(e_rev, parent_group->mem);

	output_state->set_all(0.0);
	output_state->saxpy(A_ampa,g_ampa);
	output_state->saxpy(A_nmda,g_nmda);
	output_state->mul(temp);

	g_nmda->saxpy(-mul_nmda, g_nmda);
	g_nmda->saxpy(mul_nmda, g_ampa);
	g_ampa->scale(scale_ampa);
}

void LinearComboSynapse::set_ampa_nmda_ratio(AurynFloat ratio) 
{
 	A_ampa = ratio/(ratio+1.0);
	A_nmda = 1./(ratio+1.0);
}

void LinearComboSynapse::set_nmda_ampa_current_ampl_ratio(AurynFloat ratio) 
{
	const double tau_r = tau_ampa;
	const double tau_d = tau_nmda;

	// compute amplitude of Combo conductance
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
