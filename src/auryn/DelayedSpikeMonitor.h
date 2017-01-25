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

#ifndef DELAYEDSPIKEMONITOR_H_
#define DELAYEDSPIKEMONITOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "SpikingGroup.h"
#include "Monitor.h"
#include "System.h"
#include <fstream>

namespace auryn {

/*! \brief SpikeMonitor that reads the delayed spikes as they are
 *         received by a postsynaptic neuron.
 *
 * Usually SpikeMonitor writes spikes to file for the rank that it runs on. i.e. each rank 
 * writes is own spk file which then need to be merged. This monitor writes all spikes from 
 * all ranks to files on all ranks. The spikes writen by this monitor are delayed by the 
 * axonal delay (because they need to be communicated from all ranks to all ranks first).
 * The main role of this monitor is to test SyncBuffer and Auryn's spike synchornization.
 * It records all
 * the spikes on each node (which effectively multiplies spikes).
 */
class DelayedSpikeMonitor : Monitor
{
private:
    NeuronID n_from;
    NeuronID n_to;
	SpikeContainer::const_iterator it;
	SpikingGroup * src;
	NeuronID offset;
	void init(SpikingGroup * source, std::string filename, NeuronID from, NeuronID to);
	void free();
	
public:
	DelayedSpikeMonitor(SpikingGroup * source, std::string filename);
	DelayedSpikeMonitor(SpikingGroup * source, std::string filename, NeuronID to);
	DelayedSpikeMonitor(SpikingGroup * source, std::string filename, NeuronID from, NeuronID to);
	void set_offset(NeuronID of);
	virtual ~DelayedSpikeMonitor();
	void execute();
};

}

#endif /*DELAYEDSPIKEMONITOR_H_*/
