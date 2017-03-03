/* 
* Copyright 2014-2016 Friedemann Zenke
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

#ifndef IFGROUP_H_
#define IFGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "System.h"

namespace auryn {

/*! \brief Implements the standard integrate and file model used in Auryn.
 *
 * This is the implementation of a standard leaky integrate and fire model
 * with relative refractoriness and conductance based synapses. The model
 * has two time constants for excitatory conductances (AMPA,NMDA). AMPA 
 * conductances are modelled as exponential decays. NMDA is implemented to 
 * have a double exponental timecourse, by low-pass filtering over the AMPA
 * equation. The amplitude between the individual contributions can be 
 * ajusted via set_ampa_nmda_ratio. The voltage dependence of NMDA is 
 * ignored in this model.
 */
	class IFGroup : public NeuronGroup
	{
	private:
		AurynStateVector * t_leak;
		AurynStateVector * t_exc;
		AurynStateVector * t_inh;
		AurynFloat scale_ampa,scale_gaba, scale_thr;
		AurynFloat tau_ampa,tau_gaba,tau_nmda;
		AurynFloat A_ampa,A_nmda;
		AurynFloat e_rest,e_rev,thr_rest,tau_mem,tau_thr,dthr;
		void init();
		void free();
		void calculate_scale_constants();
		void integrate_membrane();
		void integrate_linear_nmda_synapses();
		void check_thresholds();
	public:
		AurynFloat e_reset;
		/*! \brief Default constructor.
		 *
		 * @param size the size of the group.  
		 * @param distmode Node distribution mode  
		 */
		IFGroup( NeuronID size, NodeDistributionMode distmode=AUTO );
		virtual ~IFGroup();
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
		/*! Set ratio between ampa/nmda contribution to excitatory conductance. */
		void set_ampa_nmda_ratio(AurynFloat ratio);
		void clear();
		/*! Internally used evolve function. Called by System. */
		virtual void evolve();
	};
}

#endif /*IFGROUP_H_*/

