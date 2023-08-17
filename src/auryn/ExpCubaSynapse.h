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

#ifndef EXPCUBASYNAPSE_H_
#define EXPCUBASYNAPSE_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "SynapseModel.h"
#include "System.h"

namespace auryn {

/*! \brief Implements an exponential current-based synapse model
 *
 * Requires input and output state to be the same, because the model directly
 * operates on the input state, by just multiplying its value.
 *
 * Default timescale is 5e-3s.
 *
 */
	class ExpCubaSynapse : public SynapseModel
	{
	private:
		AurynFloat tau_syn;
		AurynFloat mul_syn;

	public:
		ExpCubaSynapse(NeuronGroup * parent, AurynStateVector * input, AurynStateVector * output);

		/*! \brief Sets synaptic decay time scale */
		void set_tau(const AurynState tau);

		virtual void evolve();
	};
}

#endif /*EXPCUBASYNAPSE_H_*/

