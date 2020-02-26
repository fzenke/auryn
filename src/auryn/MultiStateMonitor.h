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

#ifndef MULTISTATEMONITOR_H_
#define MULTISTATEMONITOR_H_

#define VOLTAGEMONITOR_PASTED_SPIKE_HEIGHT 20e-3

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "StateMonitor.h"
#include "System.h"
#include "Connection.h"
#include <fstream>
#include <iomanip>

namespace auryn {

/*! \brief Records a number of state variables from one or more groups to a single file
 *
 * This Monitor allows to record a number of state variables from a single or several NeuronGroups 
 * and to write them in a ASCII file that is organized column-wise. Note that this Monitor produces 
 * huge amounts of data and slows down simulations considerably. It should only be used when you know
 * what you ware doing.*/
class MultiStateMonitor : public StateMonitor
{
private:
	NeuronID gid;

protected:

	
public:
	/*! \brief List to holds pointers to the state variables to record from */
	std::vector<AurynState*> state_list;

	MultiStateMonitor(std::string filename,  AurynDouble stepsize=auryn_timestep);
	virtual ~MultiStateMonitor();

	/*! \brief Add single target variable to state list */
	void add_target(AurynState * target);

	/*! \brief Adds all elements of a given state vector to the list of monitored variables. */
	void add_neuron_range(AurynStateVector * source);

	/*! \brief Adds a range of elements to the list of monitored variables. */
	void add_neuron_range(AurynStateVector * source, NeuronID from, NeuronID to);

	/*! \brief Adds all elements of the source NeuronGroup and corresponding statename to the list of monitored variables. */
	void add_neuron_range(NeuronGroup * source, std::string statename="mem");

	/*! \brief Adds a range of elements of the source NeuronGroup and corresponding statename to the list of monitored variables. */
	void add_neuron_range(NeuronGroup * source, std::string statename, NeuronID from, NeuronID to);

	void execute();
};

}

#endif /*MULTISTATEMONITOR_H_*/
