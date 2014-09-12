/* 
* Copyright 2014 Friedemann Zenke
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
*/

#ifndef STATEMONITOR_H_
#define STATEMONITOR_H_

#include "auryn_definitions.h"
#include "Monitor.h"
#include "System.h"
#include "Connection.h"
#include <fstream>
#include <iomanip>

using namespace std;

/*! \brief Records the membrane potential from one unit from the source neuron group to a file.*/
class StateMonitor : protected Monitor
{
protected:
	/*! The source neuron group to record from */
	NeuronGroup * src;

	/*! Target variable */
	AurynState * target_variable;

	/*! The source neuron id to record from */
	NeuronID nid;
	/*! The step size (sampling interval) in units of dt */
	AurynTime ssize;
	/*! Standard initialization */
	void init(NeuronGroup * source, NeuronID id, string statename, string filename, AurynTime stepsize);
	
public:
	StateMonitor(NeuronGroup * source, NeuronID id, string statename, string filename, AurynDouble sampling_interval=dt);
	virtual ~StateMonitor();
	void propagate();
};

#endif /*STATEMONITOR_H_*/
