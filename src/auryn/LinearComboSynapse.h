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

#ifndef LINEARCOMBOSYNAPSE_H_
#define LINEARCOMBOSYNAPSE_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "SynapseModel.h"
#include "System.h"

namespace auryn {

/*! \brief Implements Auryn's default conductance based AMPA, 
 * Combo synapse without Combo voltage dependence
 *
 * Default timescales are tau_ampa = 5e-3 and tau_nmda = 100e-3.
 *
 */
	class LinearComboSynapse : public SynapseModel
	{
	private:
		AurynFloat tau_ampa, tau_nmda;
		AurynFloat scale_ampa, mul_nmda;
		AurynFloat A_ampa, A_nmda;
		AurynFloat e_rev; //!< reversal potenital

		AurynVectorFloat * g_ampa;
		AurynVectorFloat * g_nmda;
		AurynVectorFloat * temp;

	public:
		LinearComboSynapse(NeuronGroup * parent, AurynStateVector * input, AurynStateVector * output);

		virtual ~LinearComboSynapse();

		/*! \brief Sets AMPA decay time scale */
		void set_tau_ampa(const AurynState tau);

		/*! \brief Sets Combo decay time scale */
		void set_tau_nmda(const AurynState tau);

		/*! \brief Sets reversal potential */
		void set_e_rev(const AurynState reversal_pot);

		/*! \brief Set ratio between ampa/nmda contribution to excitatory conductance. 
		 *
		 * This sets the ratio between the integrals between the conductance kernels. */
		void set_ampa_nmda_ratio(AurynFloat ratio);

		/*! \brief Sets nmda-ampa amplitude ratio
		 *
		 * This sets the ratio between the amplitudes of nmda to ampa. */
		void set_nmda_ampa_current_ampl_ratio(AurynFloat ratio);

		void clear();
		virtual void evolve();
	};
}

#endif /*LINEARCOMBOSYNAPSE_H_*/

