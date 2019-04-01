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

#ifndef EXPCOBASYNAPSE_H_
#define EXPCOBASYNAPSE_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "SynapseModel.h"
#include "System.h"

namespace auryn {

/*! \brief Implements an exponential conductance-based synapse model
 *
 * Default timescale is 5e-3s.
 * We interpret input state as the neurotransmitter concentration which decays exponentially.
 * The output state is interpreted as a current which results from multiplying the potential difference
 * with the neurotransmitter concentration of input state.
 *
 * Note that output state is overwritten in each round and needs to be added to the membrane potential in 
 * the parents evolve function.
 *
 */
	class ExpCobaSynapse : public SynapseModel
	{
	private:
		AurynFloat tau_syn;
		AurynFloat mul_syn;
		AurynFloat e_rev; //!< reversal potenital

	public:
		ExpCobaSynapse(NeuronGroup * parent, AurynStateVector * input, AurynStateVector * output);
		virtual ~ExpCobaSynapse();

		/*! \brief Sets synaptic decay time scale */
		void set_tau(const AurynState tau);

		/*! \brief Sets reversal potential */
		void set_e_rev(const AurynState reversal_pot);

		virtual void evolve();
	};
}

#endif /*EXPCOBASYNAPSE_H_*/

