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

#ifndef IZHIKEVICHGROUP_H_
#define IZHIKEVICHGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "System.h"

namespace auryn {

/*! \brief This NeuronGroup implements the Izhikevich neuron model with conductance based AMPA and GABA synapses
 *
 * This NeuronGroup implements the nonlinear integrate and fire neuron model by Eugene M. Izhikevich as described in 
 * Izhikevich, E.M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks 14, 1569–1572.
 *
 * In this implementation the state variables have been rescaled to V and seconds for consistency. The model is controlled 
 * via the public members: avar, bvar, cvar, dvar which correspond to the a,b,c and d parameters in the paper.
 *
 * Note that since this model has been rescaled to volts and seconds the reset and jump size parameters (cvar and dvar)
 * also need to be given in volts (i.e. scaled by 1e-3).
 *
 * Also keep in mind that the default interpretation of synaptic synaptic strength given in multiples of the leak
 * conductance is not longer true for this group and you will need to "gauge" synaptic strength according to the size of 
 * membrane potential deflection it causes.
 *
 */
class IzhikevichGroup : public NeuronGroup
{
private:
	AurynStateVector * adaptation_vector;
	AurynStateVector * temp_vector;
	AurynStateVector * cur_exc, * cur_inh;

	AurynFloat e_rev_gaba,thr;
	AurynFloat tau_ampa,tau_gaba;
	AurynFloat scale_ampa, scale_gaba;

	void init();
	void calculate_scale_constants();
	inline void integrate_state();
	inline void check_thresholds();
	virtual string get_output_line(NeuronID i);
	virtual void load_input_line(NeuronID i, const char * buf);

	void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version );
	void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version );
public:
	/*! \brief The default constructor of this NeuronGroup */
	IzhikevichGroup(NeuronID size);
	virtual ~IzhikevichGroup();

	AurynFloat avar; /*!< The "a" parameter in the Izhikevich model which controls the time course of the adaptation variable u */
	AurynFloat bvar; /*!< The "b" parameter in the Izhikevich model which controls the fixed point of the adaptation variable u */
	AurynFloat cvar; /*!< The "c" parameter in the Izhikevich model which is the reset voltage */
	AurynFloat dvar; /*!< The "d" parameter in the Integrates model which is the spike triggered jump size of the adaptation variable u 
					   (note that it needs to be rescaled to units of V (i.e. multiplied by 1e-3) when paramters from Izhikevich's
					   original publication are used because the model has been renormalized to volts for consistency reasons. */

	/*! \brief Sets the exponential time constant for the AMPA channel (default 5ms) */
	void set_tau_ampa(AurynFloat tau);

	/*! \brief Gets the exponential time constant for the AMPA channel */
	AurynFloat get_tau_ampa();

	/*! \brief Sets the exponential time constant for the GABA channel (default 10ms) */
	void set_tau_gaba(AurynFloat tau);

	/*! \brief Gets the exponential time constant for the GABA channel */
	AurynFloat get_tau_gaba();

	/*! \brief Resets all neurons to defined and identical initial state. */
	void clear();

	/*! \brief Integrates the NeuronGroup state
	 *
	 * The evolve method internally used by System. */
	void evolve();
};

}

#endif /*IZHIKEVICHGROUP_H_*/

