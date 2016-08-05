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

#ifndef FILEINPUTGROUP_H_
#define FILEINPUTGROUP_H_

#include <fstream>
#include <sstream>

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "SpikingGroup.h"

namespace auryn {

/*! \brief Reads spikes from a ras file and emits them as SpikingGroup in a simulation.
 *
 * FileInputGroup first reads the entire ras file into memory and emits the spikes then during the simulation
 * without file access. It supports looping over the input spikes by setting the loop argument with the constructor 
 * to true.
 */
class FileInputGroup : public SpikingGroup
{
private:
	AurynTime next_event_time;
	NeuronID next_event_spike;
	bool playinloop;

	/*! \brief Aligns looped file input to a grid of this size */
	AurynTime loop_grid_size;

	AurynTime time_delay;
	AurynTime time_offset;
	AurynTime reset_time;


	std::vector<SpikeEvent_type> input_spikes;
	std::vector<SpikeEvent_type>::const_iterator spike_iter; 

	void init(string filename );

	AurynTime get_offset_clock();
	AurynTime get_next_grid_point( AurynTime time );

public:

	FileInputGroup(NeuronID n, string filename );
	FileInputGroup(NeuronID n, string filename , bool loop=true, AurynFloat delay=0.0 );
	virtual ~FileInputGroup();
	virtual void evolve();

	/*!\brief Aligned loop blocks to a temporal grid of this size 
	 *
	 * */
	void set_loop_grid(AurynDouble grid_size);

	/*!\brief Load spikes from file
	 *
	 * */
	void load_spikes(std::string filename);
};

}

#endif /*FILEINPUTROUP_H_*/
