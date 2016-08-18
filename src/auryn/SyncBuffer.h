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


#ifndef SYNCBUFFER_H_
#define SYNCBUFFER_H_


#define SYNCBUFFER_SIZE_MARGIN_MULTIPLIER 3 //!< Safety margin for receive buffer size -- a value of 3 should make overflows rare in AI state
#define SYNCBUFFER_SIZE_HIST_LEN 512 //!< Accumulate history over this number of timesteps before updating the sendbuffer size in the absence of overflows

/*! \brief Datatype used for delta computation should be a "long" for large nets with sparse activity otherwise NeuronID 
 *
 * To strictly guarnatee flawless function this datatype needs to be larger than max(NeuronID)*MINDELAY to avoid an overflow and undefined
 * behavior. */
#define SYNCBUFFER_DELTA_DATATYPE NeuronID 

#include "auryn_definitions.h"
#include "SpikeDelay.h"
#include <vector>
#include <algorithm>

#ifdef AURYN_CODE_USE_MPI

#include <boost/mpi.hpp>
#include <mpi.h>

namespace auryn {

/*! \brief Buffer object to capsulate native MPI_Allgather for SpikingGroups
 *
 * The class stores the recent history of transmission buffer sizes and tries
 * to determine an optimal size on-line.
 * */
	class SyncBuffer
	{
		private:
			std::vector<NeuronID> send_buf;
			std::vector<NeuronID> recv_buf;

			NeuronID overflow_value;

			SYNCBUFFER_DELTA_DATATYPE max_delta_size;
			SYNCBUFFER_DELTA_DATATYPE undefined_delta_size;


			// NeuronID size_history[SYNCBUFFER_SIZE_HIST_LEN];
			NeuronID maxSendSum;
			NeuronID maxSendSum2;
			NeuronID syncCount;

			/*! \brief The send buffer size that all ranks agree upon */
			NeuronID max_send_size;

			mpi::communicator * mpicom;


			SYNCBUFFER_DELTA_DATATYPE carry_offset;

			SYNCBUFFER_DELTA_DATATYPE * pop_delta_spikes;
			SYNCBUFFER_DELTA_DATATYPE * last_spike_pos;

			/*! \brief vector with offset values to allow to pop more than one delay */
			NeuronID * pop_offsets;

			void reset_send_buffer();

			void init();
			void free();

			/*! \brief Reads the next spike delta */
			NeuronID * read_delta_spike_from_buffer(NeuronID * iter, SYNCBUFFER_DELTA_DATATYPE & delta);

			/*! \brief Reads a single spike attribute */
			NeuronID * read_attribute_from_buffer(NeuronID * iter, AurynFloat & attrib);
		public:

			/*! \brief The default contructor. */
			SyncBuffer( mpi::communicator * com );

			/*! \brief The default destructor. */
			virtual ~SyncBuffer( );

			/*! \brief Synchronize spikes and additional information across ranks. */
			void sync();

			/*! \brief Pushes a spike delay with all its spikes to the SyncBuffer. */
			void push(SpikeDelay * delay, const NeuronID size);

			/*! \brief Terminate send buffer. */
			void null_terminate_send_buffer();

			/*! \brief Rerieves a spike delay with all its spikes from the SyncBuffer. */
			void pop(SpikeDelay * delay, const NeuronID size);

			/*! \brief Return max_send_size value which determines the size of the MPI AllGather operation. */
			int get_max_send_buffer_size();	

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
			AurynDouble deltaT;
			AurynDouble measurement_start;
			AurynDouble get_relative_sync_time();
			AurynDouble get_sync_time();
			AurynDouble get_elapsed_wall_time();
			void reset_sync_time();
#endif
	};
}

#endif // AURYN_CODE_USE_MPI

#endif /*SYNCBUFFER_H_*/

