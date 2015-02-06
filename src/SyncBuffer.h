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

#ifndef SYNCBUFFER_H_
#define SYNCBUFFER_H_

#define SYNCBUFFER_SIZE_MARGIN_MULTIPLIER 3 //!< Safety margin for receive buffer size -- a value of 3 should make overflows rare in AI state
#define SYNCBUFFER_SIZE_HIST_LEN 512 //!< Accumulate history over this number of timesteps before updating the sendbuffer size in the absence of overflows

#include "auryn_definitions.h"
#include "SpikeDelay.h"
#include <vector>
#include <boost/mpi.hpp>
#include <boost/progress.hpp>
#include <mpi.h>

using namespace std;


/*! \brief Buffer object to capsulate native MPI_Allgather for SpikingGroups
 *
 * The class stores the recent history of transmission buffer sizes and tries
 * to determine an optimal size on-line.
 * */

class SyncBuffer
{
	private:
		vector<NeuronID> send_buf;
		vector<NeuronID> recv_buf;

		// NeuronID size_history[SYNCBUFFER_SIZE_HIST_LEN];
		NeuronID maxSendSum;
		NeuronID maxSendSum2;
		NeuronID syncCount;

		/*! The send buffer size that all ranks agree upon */
		NeuronID max_send_size;

		mpi::communicator * mpicom;

		NeuronID overflow_value;

		NeuronID groupPushOffset1;
		NeuronID groupPopOffset;

		NeuronID count[MINDELAY]; // needed to decode attributes

		/*! vector with offset values to allow to pop more than one delay */
		vector<NeuronID> pop_offsets;

		void reset_send_buffer();

		void init();
		void free();

	public:

		SyncBuffer( mpi::communicator * com );

		void sync();

		void push(SpikeDelay * delay, NeuronID size);
		void pop(SpikeDelay * delay, NeuronID size);

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
		AurynDouble deltaT;
		AurynDouble measurement_start;
		AurynDouble get_relative_sync_time();
		AurynDouble get_sync_time();
		AurynDouble get_elapsed_wall_time();
		void reset_sync_time();
#endif
};

#endif /*SYNCBUFFER_H_*/
