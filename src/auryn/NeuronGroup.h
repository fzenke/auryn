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

#ifndef NEURONGROUP_H_
#define NEURONGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "SpikingGroup.h"

#include <map>

#include <fstream>
#include <sstream>

namespace auryn {

/*! \brief Abstract base class for all neuron groups.
 *
 * This class constitutes the abstract forefather of all neuron groups in the simulator. It serves as target for Connection objects and is directly derived from SpikingGroup. It directly allocated the memory for mem (membrane potential) and the synaptic conductantes (g_ampa, g_gaba, g_nmda) as well as a vector to store the thresholds.
 * The detailed implementation of the evolve() function depends on the children of NeuronGroup.
 */
class NeuronGroup : public SpikingGroup
{
protected:


	AurynStateVector * default_exc_target_state;
	AurynStateVector * default_inh_target_state;

	/*! Init procedure called by default constructor. */
	void init();

	/*! Called by default destructor */
	void free();




public:
	/*! Stores the membrane potentials. */
	AurynStateVector * mem ;
	/*! Stores the AMPA conductances of each point neuron. */
	AurynStateVector * g_ampa ;
	/*! Stores the GABA conductances of each point neuron. */
	AurynStateVector * g_gaba ;
	/*! Stores the NMDA conductances of each point neuron. */
	AurynStateVector * g_nmda ;

	/*! Stores  threshold terms for moving thresholds. 
	 *
	 * TODO Remove and move to downstream. */
	AurynStateVector * thr ;

	/*! \brief Default constructor */
	NeuronGroup( NeuronID n, NodeDistributionMode mode=AUTO );

	/*! \brief Old default constructor 
	 * \deprecated */
	NeuronGroup( NeuronID n, double loadmultiplier, NeuronID total );

	/*! Default destructor */
	virtual ~NeuronGroup();

	virtual void clear() = 0;

	/*! \brief Conveniently sets all values in a state vector identified by name in this group 
	 *
	 * This sets all values in a state vector identified by its string name.
	 * All string identified operations are slow, so do not use them in evolve or propagate functions, 
	 * but only for initialization.
	 *
	 * \param name The state name to address
	 * \param val The value to set all entries to 
	 * */
	void set_state(std::string name, AurynState val);

	/*! \brief Conveniently sets a single value of element i in a state vector identified by name in this group 
	 *
	 * \param name The state name to address
	 * \param i The local neuron id to set (this needs to be a translated value for MPI simulations)
	 * \param val The value to set all entries to. 
	 * */
	void set_state(std::string name, NeuronID i, AurynState val);

	/*! \brief Returns default exc target input */
	virtual AurynStateVector * get_default_exc_target();

	/*! \brief Returns default inh target input */
	virtual AurynStateVector * get_default_inh_target();


	void random_mem(AurynState mean=-60e-3, AurynState sigma=5e-3);

	virtual void init_state();

	void safe_tadd(NeuronID id, AurynWeight amount, TransmitterType t=GLUT);
	/*! Adds given transmitter to neuron as from a synaptic event. DEPRECATED. Moving slowly to SparseConnection transmit. */
	void tadd(NeuronID id, AurynWeight amount, TransmitterType t=GLUT);

	/*! Adds given amount of transmitter to neuron state/id. */
	void tadd(AurynStateVector * state, NeuronID id, AurynWeight amount);

};

}

#endif /*NEURONGROUP_H_*/
