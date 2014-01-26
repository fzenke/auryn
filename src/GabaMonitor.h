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

#ifndef GABAMONITOR_H_
#define GABAMONITOR_H_

#include "auryn_definitions.h"
#include "Monitor.h"
#include "System.h"
#include "Connection.h"
#include <fstream>
#include <iomanip>

using namespace std;

/*! \brief Records the GABA conductance from one specific unit from the source group. */
class GabaMonitor : protected Monitor
{
protected:
	NeuronGroup * src;
	NeuronID nid;
	AurynTime ssize;
	void init(NeuronGroup * source, NeuronID id, string filename, AurynTime stepsize);
	
public:
	GabaMonitor(NeuronGroup * source, NeuronID id, string filename, AurynTime stepsize=1);
	virtual ~GabaMonitor();
	void propagate();
};

#endif /*GABAMONITOR_H_*/
