/* 
* Copyright 2014-2020 Friedemann Zenke
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

#ifndef CURRENTINJECTOR_H_
#define CURRENTINJECTOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "Logger.h"
#include "Device.h"
#include "NeuronGroup.h"


namespace auryn {

/*! \brief Stimulator class to add values in each timestep to arbitrary neuronal states. 
 *
 * Most commonly used to inject "currents" to arbitraty neuronal states. Maintains an internal vector with
 * numbers which are added (times auryn_timestep) in each timestep to the neuronal target vector 
 * (per default that is the membrane voltage and hence the operation corresponds to injecting a current).
 * Note that because of this current units of CurrentInjector are in a sense arbitrary because they depend 
 * on the neuron model.
 * 
 */

class CurrentInjector : protected Device
{
private:



	void free();

	/*! \brief Returns the lambda parameter of the pmf for Current. */
	AurynFloat get_lambda();

	/*! \brief Scale factor which should include auryn_timestep and any respective resistance. */
	AurynFloat alpha;

protected:

	/*! \brief Target membrane */
	AurynVectorFloat * target_vector;

	/*! \brief Vector storing all the current values */
	AurynVectorFloat * currents;

	/*! \brief The target NeuronGroup */
	NeuronGroup * dst;

	
public:

	/*! \brief Default Constructor 
	 * @param[target] The target group
	 * @param[neuron_state_name] The state to manipulate
	 * @param[initial_current] Initializes all currents with this value
	 */
	CurrentInjector(NeuronGroup * target, std::string neuron_state_name="mem", AurynFloat initial_current=0.0 );

	/*! \brief Sets the state to add the "current" in every timestep to */
	void set_target_state( std::string state_name = "mem" );

	/*! \brief Default Destructor */
	virtual ~CurrentInjector();

	/*! \brief Sets current strengh for neuron i
	 * 
	 * This must be a valid state vector name (default = mem) 
	 * 
	 * \param i Index of neuron
	 * \param current Current value to set*/
	void set_current( NeuronID i, AurynFloat current );

	/*! \brief Sets current strength for all neurons
	 * 
	 * This must be a valid state vector name (default = mem) 
	 * 
	 * \param current Current value to set*/
	void set_all_currents( AurynFloat current );

	/*! \brief Sets scale 
	 *
	 * \param scale The scale value to set
	 *
	 * This sets a scaling factor or "unit" with which all current values are
	 * multiplied. Internally this value is multiplied by auryn time step to
	 * make it invariant to timestep changes.
	 * */
	void set_scale( AurynFloat scale );

	/*! Implementation of necessary propagate() function. */
	void execute();



};

}

#endif /*CURRENTINJECTOR_H_*/
