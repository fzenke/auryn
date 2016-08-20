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


#include "SyncBuffer.h"

#ifdef AURYN_CODE_USE_MPI
using namespace auryn;

SyncBuffer::SyncBuffer( mpi::communicator * com )
{
	mpicom = com;
	init();
}

SyncBuffer::~SyncBuffer(  )
{
	delete [] pop_offsets;
	delete [] pop_delta_spikes;
	delete [] last_spike_pos;
}

void SyncBuffer::init()
{

	/* Overflow value for syncbuffer */
	overflow_value = std::numeric_limits<NeuronID>::max();

	/* Maximum delta size. We make this one smaller than max to avoid problems 
	 * if the datatype is the same as NeuronID and then the max corresponds
	 * to the overflow value in send buffer ... */
	max_delta_size = overflow_value-1; 

	/* Overflow value for delta value. Should be different from the ones above. */
	undefined_delta_size = overflow_value-2; 

	pop_offsets = new NeuronID[mpicom->size()];
	pop_delta_spikes = new SYNCBUFFER_DELTA_DATATYPE[ mpicom->size() ];
	last_spike_pos = new SYNCBUFFER_DELTA_DATATYPE[ mpicom->size() ];
	for ( int i = 0 ; i < mpicom->size() ; ++i ) { 
		pop_offsets[i] = 0;
		pop_delta_spikes[i] = undefined_delta_size;
		last_spike_pos[i] = 0;
	}


	maxSendSum = 0;
	maxSendSum2 = 0;
	syncCount = 0;

	max_send_size = 4;
	recv_buf.resize(mpicom->size()*max_send_size);

	reset_send_buffer();


#ifdef CODE_COLLECT_SYNC_TIMING_STATS
    measurement_start = MPI_Wtime();     
	deltaT = 0.0;
#endif

}

void SyncBuffer::push(SpikeDelay * delay, const NeuronID size)
{
	// DEBUG
	// std::cout << "Rank " << mpicom->rank() << "push\n";
	// delay->print();
	
	const SYNCBUFFER_DELTA_DATATYPE grid_size = (SYNCBUFFER_DELTA_DATATYPE)size*MINDELAY;

	SYNCBUFFER_DELTA_DATATYPE unrolled_last_pos = 0;
	// circular loop over different delay bins
	for (int slice = 0 ; slice < MINDELAY ; ++slice ) {
		SpikeContainer * sc = delay->get_spikes(slice+1);
		AttributeContainer * ac = delay->get_attributes(slice+1);

		// loop over all spikes in current delay time slice
		for (int i = 0 ; 
				i < sc->size() ; 
				++i ) {
			NeuronID spike = sc->at(i);
			// compute unrolled position in current delay
			SYNCBUFFER_DELTA_DATATYPE unrolled_pos = (SYNCBUFFER_DELTA_DATATYPE)(spike) + (SYNCBUFFER_DELTA_DATATYPE)size*slice; 
			// compute vertical unrolled difference from last spike
			SYNCBUFFER_DELTA_DATATYPE spike_delta = unrolled_pos + carry_offset - unrolled_last_pos;
			// memorize current position in slice
			unrolled_last_pos = unrolled_pos; 
			// discard carry_offset since its only added to the first spike_delta
			carry_offset = 0;

			// overflow managment -- should only ever kick in for very very large SpikingGroups and very very sparse activity
			while ( spike_delta >= max_delta_size ) {
				send_buf.push_back( max_delta_size );
				spike_delta -= max_delta_size;
			}
		
			// storing the spike delta (or its remainder) to buffer
			send_buf.push_back(spike_delta);

			// append spike attributes here in buffer
			for ( int k = 0 ; k < delay->get_num_attributes() ; ++k ) { // loop over attributes
				NeuronID cast_attrib = *(NeuronID*)(&(ac->at(i*delay->get_num_attributes()+k)));
				send_buf.push_back(cast_attrib);
				// std::cout << "store " << std::scientific << ac->at(i*delay->get_num_attributes()+k) << " int " << cast_attrib << std::endl;
			}
		}
	}

	// set save carry_offset which is the remaining difference from the present group
	// plus because there might be more than one group without a spike ...
	carry_offset += grid_size-unrolled_last_pos;

}


void SyncBuffer::null_terminate_send_buffer()
{
	// puts a "delta spike" just behind the last unrolled delay of the last group
	// std::cout << " term " << carry_offset << std::endl;
	send_buf.push_back(carry_offset);
}

NeuronID * SyncBuffer::read_delta_spike_from_buffer(NeuronID * iter, SYNCBUFFER_DELTA_DATATYPE & delta)
{
	delta = 0;

	// add overflow packages if there are any 
	while ( *iter == max_delta_size ) {
		delta += *iter;
		iter++; 
	}

	// adds element which is not an overflow pacakge
	delta += *iter;

	iter++; 
	return iter;
}

NeuronID * SyncBuffer::read_attribute_from_buffer(NeuronID * iter, AurynFloat & attrib)
{
	attrib = *((AurynFloat*)(iter));
	iter++; 
	return iter;
}

void SyncBuffer::pop(SpikeDelay * delay, const NeuronID size)
{
	// TODO consider passing the current rank, because in principle it should not require the sync
	
	const SYNCBUFFER_DELTA_DATATYPE grid_size = (SYNCBUFFER_DELTA_DATATYPE)size*MINDELAY;

	// clear all receiving buffers in the relevant time range
	for (NeuronID i = 1 ; i < MINDELAY+1 ; ++i ) {
		delay->get_spikes(i)->clear();
		delay->get_attributes(i)->clear();
	}

	// loop over different rank input segments in recv_buf
	for (int r = 0 ; r < mpicom->size() ; ++r ) {

		// when we enter this function we know this is a new group
		last_spike_pos[r] = 0;

		//read current difference element from buffer and interpret as unrolled
		NeuronID * iter = &recv_buf[r*max_send_size+pop_offsets[r]]; // first spike

		while ( true ) {

			// std::cout << "have " << pop_delta_spikes[r] << std::endl;
			
			if ( pop_delta_spikes[r] == undefined_delta_size ) {
				// read delta spike value from buffer and update iterator
				iter = read_delta_spike_from_buffer(iter, pop_delta_spikes[r]);
			}

			SYNCBUFFER_DELTA_DATATYPE unrolled_spike = last_spike_pos[r] + pop_delta_spikes[r];
			if ( unrolled_spike < grid_size ) {

				// decode spike positon on time slice grid
				const int slice = unrolled_spike/size;
				const NeuronID spike = unrolled_spike%size;
				
				// push spike to appropriate time slice
				delay->get_spikes(slice+1)->push_back(spike);
				// std::cout << "slice " << slice << " spike " << spike << std::endl;

				// save last position
				last_spike_pos[r] = unrolled_spike;
				pop_delta_spikes[r] = undefined_delta_size;

				// now that we know where the spike belongs we read the spike attributes from the buffer
				for ( int k = 0 ; k < delay->get_num_attributes() ; ++k ) { // loop over attributes
					AurynFloat attrib;
					iter = read_attribute_from_buffer(iter, attrib);
					delay->get_attributes(slice+1)->push_back(attrib);
					// std::cout << "read " << std::scientific << attrib << std::endl;
				}

			} else {
				pop_delta_spikes[r] -= (grid_size-last_spike_pos[r]);
				// std::cout << "keep " << pop_delta_spikes[r] << std::endl;
				break;
			}
		}
		pop_offsets[r] = iter - &recv_buf[r*max_send_size]; // save offset in recv_buf section
	}

	// // TEST TODO comment after testing
	// for (NeuronID i = 1 ; i < MINDELAY+1 ; ++i ) {
	// 	SpikeContainer * myvector = delay->get_spikes(i);
	// 	std::sort (myvector->begin(), myvector->end());
	// }

#ifdef DEBUG
	if ( mpicom->rank() == 0 ) {
		for ( NeuronID slice = 0 ; slice < MINDELAY ; ++slice ) {
			if ( delay->get_attributes(slice+1)->size() != delay->get_num_attributes()*delay->get_spikes(slice+1)->size() ) {
				std::cout << "   " << delay->get_spikes(slice+1)->size() << " spikes extracted in time slice " << slice+1 << std::endl
					<< "   " << delay->get_attributes(slice+1)->size() << " attributes extracted in time slice " << slice+1
					<< std::endl;
			}
		}
	}
#endif // DEBUG
}


void SyncBuffer::sync() 
{
	if ( syncCount >= SYNCBUFFER_SIZE_HIST_LEN ) {  // update the estimate of maximum send size
		NeuronID mean_send_size =  maxSendSum/syncCount; 
		NeuronID var_send_size  =  (maxSendSum2-mean_send_size*mean_send_size)/syncCount;
		NeuronID upper_estimate =  mean_send_size+SYNCBUFFER_SIZE_MARGIN_MULTIPLIER*sqrt(var_send_size);

		if ( max_send_size > upper_estimate && max_send_size > 4 ) { 
			max_send_size = (max_send_size+upper_estimate)/2;
			recv_buf.resize(mpicom->size()*max_send_size);
#ifdef DEBUG
			std::cerr << "Reducing maximum send buffer size to "
				<< max_send_size
				<< std::endl;
#endif //DEBUG
		}	
		maxSendSum = 0;
		maxSendSum2 = 0;
		syncCount = 0;
	}

	int ierr = 0;

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	double T1, T2;              
    T1 = MPI_Wtime();     /* start time */
#endif
	if ( send_buf.size() <= max_send_size ) {
		ierr = MPI_Allgather(send_buf.data(), send_buf.size(), MPI_UNSIGNED, 
				recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
	} else { 
		// Create an overflow package 
		// std::cout << " overflow " << overflow_value << " " << send_buf.size() << std::endl;
		NeuronID overflow_data [2]; 
		overflow_data[0] = overflow_value;
		overflow_data[1] = send_buf.size(); 
		ierr = MPI_Allgather(&overflow_data, 2, MPI_UNSIGNED, 
				recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
	}


	// error handling
	if ( ierr ) {
		std::cerr << "Error during MPI_Allgather." << std::endl;
		// TODO add an exception to actually break the run here
	}

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
    T2 = MPI_Wtime();     /* end time */
	deltaT += (T2-T1);
#endif

	/* Detect overflow */
	bool overflow = false;
	NeuronID new_send_size = 0;
	for (int r = 0 ; r < mpicom->size() ; ++r ) {
		if  ( recv_buf[r*max_send_size]==overflow_value ) {
			overflow = true;
			NeuronID value = recv_buf[r*max_send_size+1];
			if ( value > new_send_size ) {
				new_send_size = value;
			}
		}
	}


	if ( overflow ) {
#ifdef DEBUG
		std::cerr << "Overflow in SyncBuffer adapting buffersize to "
			<< (new_send_size+1)*sizeof(NeuronID)
			<< " ( "
			<< mpicom->size()*(new_send_size+1)*sizeof(NeuronID)
			<< " total ) " 
			<< std::endl;
#endif //DEBUG
		max_send_size = new_send_size+2;
		recv_buf.resize(mpicom->size()*max_send_size);
		// resend full buffer
		ierr = MPI_Allgather(send_buf.data(), send_buf.size(), MPI_UNSIGNED, 
		 		recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
	} 

	// reset
	NeuronID largest_message = 0;
	for ( int i = 0 ; i < mpicom->size() ; ++i ) { 
		largest_message = std::max(pop_offsets[i],largest_message);
	}
	maxSendSum += largest_message;
	maxSendSum2 += largest_message*largest_message;


	syncCount++;

	reset_send_buffer();
}

void SyncBuffer::reset_send_buffer()
{
	send_buf.clear();

	// reset carry offsets for push and pop functions
	carry_offset = 0;
	for ( int i = 0 ; i < mpicom->size() ; ++i ) { 
		pop_offsets[i] = 0;
		pop_delta_spikes[i] = undefined_delta_size;
	}
}

int SyncBuffer::get_max_send_buffer_size()
{
	return max_send_size;
}

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
AurynDouble SyncBuffer::get_relative_sync_time()
{
	AurynDouble temp = MPI_Wtime();
	return (deltaT/(temp-measurement_start));
}

AurynDouble SyncBuffer::get_sync_time()
{
	return deltaT;
}

AurynDouble SyncBuffer::get_elapsed_wall_time()
{
	AurynDouble temp = MPI_Wtime();
	return temp-measurement_start;
}

void SyncBuffer::reset_sync_time()
{
	AurynDouble temp = MPI_Wtime();
    measurement_start = temp;     
	deltaT = 0.0;
}

#endif

#endif // AURYN_CODE_USE_MPI
