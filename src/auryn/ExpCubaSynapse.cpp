/* 
* Copyright 2014-2019 Friedemann Zenke
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

#include "ExpCubaSynapse.h"

using namespace auryn;

ExpCubaSynapse::ExpCubaSynapse(NeuronGroup * parent, AurynStateVector * input, AurynStateVector * output) 
	: SynapseModel( parent, input, output )
{
	if ( input != output ) {
		logger->warning("ExpCubaSynapse requires input and output state to be the same!");
	}

	set_tau(5e-3); // sets a default timescale
}

void ExpCubaSynapse::set_tau(const AurynState tau)
{
	tau_syn = tau;
	mul_syn = std::exp(-auryn_timestep/tau_syn);
}

void ExpCubaSynapse::evolve()
{
	input_state->scale(mul_syn);
}

