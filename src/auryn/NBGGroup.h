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

#ifndef NBGGROUP_H_
#define NBGGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "System.h"

namespace auryn {

/*! \brief This file implements NBGGroup, the Naud-Barthellier-Gerstner two compartment model 
 *
 * \version v0.1
 *
 * The model captures several key aspects of active dendrites and is described in detail in:
 * Naud, R., Bathellier, B., and Gerstner, W. (2014). Spike-timing prediction
 * in cortical neurons with active dendrites. Front. Comput. Neurosci 8, 90.
 * https://www.frontiersin.org/articles/10.3389/fncom.2014.00090/full
 *
 * */

	class NBGGroup : public NeuronGroup
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


		AurynFloat tau_ampa, tau_gaba, tau_nmda, tau_thr, tau_m, tau_wsoma, tau_wdend;
		AurynFloat A_ampa, A_nmda;
		AurynFloat e_rest, e_inh, e_thr, e_dend, e_spk_thr;
		AurynFloat e_reset;
		AurynFloat Cs, gs, Cd, gd, alpha, beta, box_height, xi; 
		AurynFloat a_wdend, b_wsoma, aux_a_wdend;

		AurynFloat mul_soma, mul_wsoma, mul_dend, mul_nmda, mul_alpha, mul_beta, mul_wdend;
		AurynFloat scale_ampa, scale_gaba, mul_thr; 
		AurynFloat scale_wsoma, scale_wdend; 

		void init();
		void free();
		void precalculate_constants();
		void integrate_membrane();
		void integrate_linear_nmda_synapses();
		void check_thresholds();

	public:
		/*! \brief Default constructor.
		 *
		 * @param size the size of the group.  
		 * @param distmode Node distribution mode  
		 */
		NBGGroup( NeuronID size, NodeDistributionMode distmode=AUTO );
		virtual ~NBGGroup();
		/*! \brief Sets the membrane time constant */
		void set_tau_mem(AurynFloat taum);
		/*! \brief Returns the membrane time constant */
		AurynFloat get_tau_mem();

		/*! \brief Sets the exponential decay time constant of the AMPA conductance (default=5ms). */
		void set_tau_ampa(AurynFloat tau);

		/*! \brief Returns the exponential decay time constant of the AMPA conductance. */
		AurynFloat get_tau_ampa();

		/*! \brief Sets the exponential decay time constant of the GABA conductance (default=10ms). */
		void set_tau_gaba(AurynFloat tau);

		/*! \brief Returns the exponential decay time constant of the GABA conductance. */
		AurynFloat get_tau_gaba();

		/*! \brief Sets the exponential decay time constant of the NMDA conductance (default=100ms).
		 *
		 * The rise is governed by tau_ampa if tau_ampa << tau_nmda. */
		void set_tau_nmda(AurynFloat tau);

		/*! \brief Sets the exponential decay time constant of the threshold (default=5).
		 *
		 * Reflects absolute and relative refractory period. */
		void set_tau_thr(AurynFloat tau);

		/*! Returns the exponential decay time constant of the NMDA conductance.
		 * The rise is governed by tau_ampa if tau_ampa << tau_nmda. */
		AurynFloat get_tau_nmda();

		/*! \brief Set ratio between ampa/nmda contribution to excitatory conductance. 
		 *
		 * This sets the ratio between the integrals between the conductance kernels. */
		void set_ampa_nmda_ratio(AurynFloat ratio);

		/*! \brief Sets nmda-ampa amplitude ratio
		 *
		 * This sets the ratio between the amplitudes of nmda to ampa. */
		void set_nmda_ampa_current_ampl_ratio(AurynFloat ratio);

		void clear();

		/*! Internally used evolve function. Called by System. */
		virtual void evolve();
	};
}

#endif /*NBGGROUP_H_*/

