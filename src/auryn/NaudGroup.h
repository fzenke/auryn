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

#ifndef NAUDGROUP_H_
#define NAUDGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "System.h"
#include "ExpCobaSynapse.h"
#include "LinearComboSynapse.h"

namespace auryn {

/*! \brief This file implements NaudGroup, Richard Naud's reduced two compartment model with active dendrites
 *
 * \version v0.1
 *
 * The model captures several key aspects of active dendrites and is a
 * reduction of an earlier model described in detail in:
 * Naud, R., Bathellier, B., and Gerstner, W. (2014). Spike-timing prediction
 * in cortical neurons with active dendrites. Front. Comput. Neurosci 8, 90.
 * https://www.frontiersin.org/articles/10.3389/fncom.2014.00090/full
 *
 * */

	class NaudGroup : public NeuronGroup
	{
	private:
		AurynStateVector * state_soma; /*!< somatic voltage */
		AurynStateVector * state_dend; /*!< dendritic voltage */
		AurynStateVector * state_wsoma; /*!< adaptation variable */
		AurynStateVector * state_wdend; /*!< activation level of putative Ca-activated potassium current */

		AurynVector<unsigned int> * post_spike_countdown;
		unsigned int post_spike_reset;

		/* auxiliary state vectors used to compute synaptic currents */
		AurynStateVector * t_leak;
		AurynStateVector * t_exc;
		AurynStateVector * t_inh;
		AurynStateVector * temp;


		AurynStateVector * g_ampa_dend, *g_gaba_dend;
		AurynStateVector * syn_current_exc_soma, *syn_current_exc_dend;
		AurynStateVector * syn_current_inh_soma, *syn_current_inh_dend;

		AurynFloat tau_ampa, tau_gaba, tau_nmda, tau_thr, tau_soma, tau_dend, tau_wsoma, tau_wdend;
		AurynFloat A_ampa, A_nmda;
		AurynFloat e_rest, e_inh, e_thr, e_dend, e_spk_thr;
		AurynFloat e_reset;
		AurynFloat Cs, gs, Cd, gd, alpha, beta, zeta, box_height, box_kernel_length, xi; 
		AurynFloat a_wdend, b_wsoma, aux_mul_wdend;

		AurynFloat mul_soma, mul_wsoma, mul_dend, mul_nmda, mul_alpha, mul_beta, mul_wdend;
		AurynFloat scale_ampa, scale_gaba, mul_thr; 
		AurynFloat scale_wsoma; 

		void init();
		void free();
		void precalculate_constants();
		void integrate_membrane();
		void integrate_linear_nmda_synapses();
		void check_thresholds();

	public:
		LinearComboSynapse * syn_exc_soma, *syn_exc_dend; // AMPA and NMDA
		ExpCobaSynapse * syn_inh_soma, *syn_inh_dend; // GABA

		/*! \brief Default constructor.
		 *
		 * @param size the size of the group.  
		 * @param distmode Node distribution mode  
		 */
		NaudGroup( NeuronID size, NodeDistributionMode distmode=AUTO );
		virtual ~NaudGroup();

		/*! \brief Sets the membrane time constant */
		void set_tau_mem(AurynFloat taum);

		/*! \brief Returns the membrane time constant */
		AurynFloat get_tau_mem();

		/*! \brief Sets the membrane time constant */
		void set_tau_thr(AurynFloat tau);

		void clear();

		/*! Internally used evolve function. Called by System. */
		virtual void evolve();
	};
}

#endif /*NAUDGROUP_H_*/

