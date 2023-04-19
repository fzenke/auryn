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

#ifndef FILEISTIMROUP_H_
#define FILESTIMGROUP_H_

#include <fstream>
#include <sstream>

#include <math.h>
#include <vector>

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "SpikingGroup.h"

namespace auryn {

/*! \brief Reads spikes from a ras file and emits them as SpikingGroup in a simulation.
 *
 * FileStimGroup first reads the entire ras file into memory and emits the spikes then during the simulation
 * without file access. It supports looping over the input spikes by setting the loop argument with the constructor 
 * to true.
 *
 */
class FileStimGroup : public SpikingGroup
{
private:
	AurynTime next_event_time;
	NeuronID next_event_spike;
	unsigned int current_pattern_index;

	/*! \brief Aligns looped file input to a grid of this size */
	AurynTime loop_grid_size;

	AurynTime time_offset;
	AurynTime reset_time;


	std::vector< std::vector<SpikeEvent_type> * > input_patterns;
	std::vector< SpikeEvent_type > * current_pattern;
	std::vector< SpikeEvent_type >::const_iterator spike_iter; 

	void init();

	void clear_input_patterns();

	AurynTime get_offset_clock();
	AurynTime get_next_grid_point( AurynTime time );


protected:

	/*! \brief Seeds the random number generator for all stimulus groups of the simulation. */
	void seed( int rndseed );

	/*! generates info for what stimulus is active. Is supposed to give the same result on all nodes (hence same seed required) */
	static boost::mt19937 order_gen; 
	static boost::uniform_01<boost::mt19937> order_die; 


public:

	/*! \brief Switch that activates the loop mode of FileStimGroup when set to true */
	bool playinloop;

	/*! \brief Stimulus order */
	StimulusGroupModeType stimulus_order ;

	/*! \brief Time delay after each replay of the input pattern in loop mode (default 0s) */
	AurynTime time_delay;

	/*! \brief Default constructor */
	FileStimGroup( NeuronID n, string filename , bool loop=false, AurynFloat delay=0.0 );

	/*! \brief Constructor which does not load spikes.
	 *
	 * Use this constructor if you want to init the group without a filename containing spikes.
	 * Spikes can be added after initialization by calling load_spikes(filename) or by adding
	 * spikes one by one using add_spike(spiketime, neuron_id). */
	FileStimGroup( NeuronID n );

	virtual ~FileStimGroup();
	virtual void evolve();


	/*! \brief Sets the stimulation mode. Can be any of StimulusGroupModeType (MANUAL,RANDOM,SEQUENTIAL,SEQUENTIAL_REV). */
	void set_stimulation_mode(StimulusGroupModeType mode);

	/*!\brief Aligned loop blocks to a temporal grid of this size 
	 *
	 * The grid is applied after the delay. It's mostly there to facilitate the
	 * synchronization of certain events or readouts.  If no grid is defined
	 * the last input spike time will define the end of the loop window (which
	 * often could be a fractional time). 
	 * */
	void set_loop_grid(AurynDouble grid_size);

	/*!\brief Load spike patterns from file
	 *
	 * */
	void load_spikes(std::string filename);


	/*!\brief Load spikes from file without erasing existing buffers.
	 *
	 * */
	void add_spikes(std::string filename);

	/*!\brief Adds a spike to the buffer manually and then temporally sorts the buffer
	 *
	 * \param spiketime the spike time 
	 * \param neuron_id the neuron you want to spike with neuron_id < size of the group
	 * */
	void add_spike(double spiketime, NeuronID neuron_id=0);

	/*!\brief Sorts spikes by time
	 *
	 * Spikes need to be sorted befure a run otherwise the behavior of this SpikingGroup is undefined
	 * */
	void sort_spikes( std::vector< SpikeEvent_type > * pattern );
};

}

#endif /*FILESTIMROUP_H_*/
