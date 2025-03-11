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

#ifndef SYNAPSEMODEL_H_
#define SYNAPSEMODEL_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "System.h"

namespace auryn {

/*! \brief Implements base class for modular synapse models
 *
 * Synapse models piggy back onto a NeuronGroup,
 * and operate on StateVectors. They typically read transmitter 
 * input from one state vector (the target state of a Connection 
 * object) and output a "current" to a target state. 
 * Input and target can be the same, for instance when the required function
 * can be achieved by simply decaying it, or they may output to another output
 * state (e.g. double filtering the input to implement an NMDA synapse) which
 * is then fed to the membrane.
 *
 * Synapse models should implement an evolve function which has to be called
 * by the parent NeuronGroup.
 * 
 */
	class SynapseModel 
	{
	protected:
		NeuronGroup * parent_group;
		AurynStateVector * input_state;
		AurynStateVector * output_state;

	public:
		SynapseModel(NeuronGroup * parent, AurynStateVector * input, AurynStateVector * output);

		virtual void evolve() = 0;
	};
}

#endif /*SYNAPSEMODEL_H_*/

