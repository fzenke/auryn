/*
* Copyright 2014-2015 Friedemann Zenke
* Contributed by Ankur Sinha
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

#ifndef BINARYSPIKEMONITOR_H_
#define BINARYSPIKEMONITOR_H_

#include "auryn_definitions.h"
#include "SpikingGroup.h"
#include "Monitor.h"
#include "System.h"
#include <fstream>

using namespace std;

/*! \brief The standard Monitor object to record spikes from a
 * SpikingGroup and write them to file
 *
 * BinarySpikeMonitor is specified with a source group of type SpikingGroup
 * and writes all or a specified range of the neurons spikes to a
 * file that has to be given at construction time.
 */
class BinarySpikeMonitor : Monitor
{
private:
    NeuronID n_from;
    NeuronID n_to;
    NeuronID n_every;
	SpikeContainer::const_iterator it;
	SpikingGroup * src;
	NeuronID offset;
	void init(SpikingGroup * source, string filename, NeuronID from, NeuronID to);
	virtual void open_output_file(string filename);
	void free();

public:
	BinarySpikeMonitor(SpikingGroup * source, string filename);
	BinarySpikeMonitor(SpikingGroup * source, string filename, NeuronID to);
	BinarySpikeMonitor(SpikingGroup * source, string filename, NeuronID from, NeuronID to);
	void set_offset(NeuronID of);
	void set_every(NeuronID every);
	virtual ~BinarySpikeMonitor();
	void propagate();
};

#endif /*BINARYSPIKEMONITOR_H_*/
