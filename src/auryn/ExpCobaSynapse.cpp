/* 
* Copyright 2014-2018 Friedemann Zenke
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

#include "ExpCobaSynapse.h"

using namespace auryn;

ExpCobaSynapse::ExpCobaSynapse(NeuronGroup * parent, AurynStateVector * input, AurynStateVector * output) 
	: SynapseModel( parent, input, output )
{
	if ( input == output ) {
		logger->warning("ExpCobaSynapse requires input and output state to be different!");
	}

	set_tau(5e-3); // sets a default timescale
	set_e_rev(0.0); // sets default reversal potential
}

ExpCobaSynapse::~ExpCobaSynapse()
{
}

void ExpCobaSynapse::set_tau(const AurynState tau)
{
	tau_syn = tau;
	mul_syn = std::exp(-auryn_timestep/tau_syn);
}

void ExpCobaSynapse::set_e_rev(const AurynState reversal_pot)
{
	e_rev = reversal_pot;
}

void ExpCobaSynapse::evolve()
{
	output_state->diff(e_rev, parent_group->mem);
	output_state->mul(input_state);
	input_state->scale(mul_syn);
}

