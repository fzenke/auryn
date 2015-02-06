/* 
* Copyright 2014-2015 Friedemann Zenke
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

#ifndef NMDAMONITOR_H_
#define NMDAMONITOR_H_

#include "auryn_definitions.h"
#include "Monitor.h"
#include "System.h"
#include "Connection.h"
#include <fstream>
#include <iomanip>

using namespace std;

/*! \brief Records the NMDA conductance from one specific unit from the source group. */
class NmdaMonitor : protected Monitor
{
protected:
	NeuronGroup * src;
	NeuronID nid;
	AurynTime ssize;
	void init(NeuronGroup * source, NeuronID id, string filename, AurynTime stepsize);
	
public:
	NmdaMonitor(NeuronGroup * source, NeuronID id, string filename, AurynTime stepsize=1);
	virtual ~NmdaMonitor();
	void propagate();
};

#endif /*NMDAMONITOR_H_*/
