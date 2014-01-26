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

#ifndef SPIKEMONITOR_H_
#define SPIKEMONITOR_H_

#include "auryn_definitions.h"
#include "SpikingGroup.h"
#include "Monitor.h"
#include "System.h"
#include <fstream>

using namespace std;

/*! \brief The standard Monitor object to record spikes from a 
 * SpikingGroup and write them to file
 *
 * SpikeMonitor is specified with a source group of type SpikingGroup
 * and writes all or a specified range of the neurons spikes to a
 * file that has to be given at construction time.
 */
class SpikeMonitor : Monitor
{
private:
    NeuronID n_from;
    NeuronID n_to;
    NeuronID n_every;
	SpikeContainer::const_iterator it;
	SpikingGroup * src;
	NeuronID offset;
	void init(SpikingGroup * source, string filename, NeuronID from, NeuronID to);
	void free();
	
public:
	SpikeMonitor(SpikingGroup * source, string filename);
	SpikeMonitor(SpikingGroup * source, string filename, NeuronID to);
	SpikeMonitor(SpikingGroup * source, string filename, NeuronID from, NeuronID to);
	void set_offset(NeuronID of);
	void set_every(NeuronID every);
	virtual ~SpikeMonitor();
	void propagate();
};

#endif /*SPIKEMONITOR_H_*/
