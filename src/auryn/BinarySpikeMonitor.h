/*
* Copyright 2014-2016 Friedemann Zenke
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
#include "AurynVector.h"
#include "SpikingGroup.h"
#include "Monitor.h"
#include "System.h"
#include <fstream>

namespace auryn {

/*! \brief The standard Monitor object to record spikes from a
 * SpikingGroup and write them to a binary file
 *
 * BinarySpikeMonitor is specified with a source group of type SpikingGroup
 * and writes all or a specified range of the neurons spikes to a
 * file that has to be given at construction time.
 * The output files can be read and converted to ascii ras files using the tool
 * aube (Auryn Binary Extractor) which compiles in the tools folder.
 */
class BinarySpikeMonitor : public Monitor
{
private:
	static const std::string default_extension;
    NeuronID n_from;
    NeuronID n_to;
    NeuronID n_every;
	SpikeContainer::const_iterator it;
	SpikingGroup * src;
	NeuronID offset;
	void init(SpikingGroup * source, std::string filename, NeuronID from, NeuronID to);
	virtual void open_output_file(std::string filename);
	void free();

public:
	BinarySpikeMonitor(SpikingGroup * source, std::string filename="");
	BinarySpikeMonitor(SpikingGroup * source, std::string filename, NeuronID to);
	BinarySpikeMonitor(SpikingGroup * source, std::string filename, NeuronID from, NeuronID to);
	void set_offset(NeuronID of);
	void set_every(NeuronID every);
	virtual ~BinarySpikeMonitor();
	virtual void execute();
	virtual void flush();
};


}

#endif /*BINARYSPIKEMONITOR_H_*/
