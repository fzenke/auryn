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

using namespace auryn;

SyncBuffer::SyncBuffer( mpi::communicator * com )
{
	mpicom = com;
	init();
}

SyncBuffer::~SyncBuffer(  )
{
	delete pop_offsets;
	delete pop_carry_offsets;
}

void SyncBuffer::init()
{

	pop_offsets = new NeuronID[mpicom->size()];
	pop_carry_offsets = new NeuronID[ mpicom->size() ];
	for ( int i = 0 ; i < mpicom->size() ; ++i ) { 
		pop_offsets[i] = 0;
		pop_carry_offsets[i] = 0;
	}

	overflow_value = -1;


	maxSendSum = 0;
	maxSendSum2 = 0;
	syncCount = 0;

	max_send_size = 2;
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
	
	AurynLong unrolled_last_pos = 0;
	bool at_least_one_spike = false;
	// circular loop over different delay bins
	for (int slice = 0 ; slice < MINDELAY ; ++slice ) {
		SpikeContainer * sc = delay->get_spikes(slice+1);

		count[slice] = 0;
		// loop over all spikes in current delay time slice
		for (SpikeContainer::const_iterator spike = sc->begin() ; 
			spike != sc->end() ; 
			++spike ) {
				// compute unrolled position in current delay
				AurynLong unrolled_pos = *spike + slice*size; 
				// compute vertical unrolled difference from last spike
				AurynLong spike_delta = unrolled_pos + carry_offset - unrolled_last_pos ;
				// memorize current position in slice
				unrolled_last_pos = unrolled_pos; 
				// discard carry_offset since its only added to the first spike_delta
				carry_offset = 0;

				// overflow managment should only ever kick in for very very large neuron groups
				while ( spike_delta >= std::numeric_limits<NeuronID>::max() ) {
					send_buf.push_back(std::numeric_limits<NeuronID>::max());
					spike_delta -= std::numeric_limits<NeuronID>::max();
					std::cout << " adding overflow package" << std::endl;
				}
				
				// std::cout << " spike " << *spike << " push_back " << spike_delta << std::endl;
				send_buf.push_back(spike_delta);
				count[slice]++;

				at_least_one_spike = true;
		}
	}

	// set save carry_offset which is the remaining difference from the present group
	// plus because there might be more than one group without a spike ...
	if ( at_least_one_spike )
		carry_offset = MINDELAY*size-unrolled_last_pos;
	else
		carry_offset += MINDELAY*size;

	// transmit attributes for count spikes for all time slices of this group
	if ( delay->get_num_attributes() ) {
		for (int i = 1 ; i < MINDELAY+1 ; ++i ) {
			AttributeContainer * ac = delay->get_attributes(i);
			for ( int k = 0 ; k < delay->get_num_attributes() ; ++k ) { // loop over attributes
				for ( NeuronID s = 0 ; s < count[i-1] ; ++s ) { // loop over spikes
					send_buf.push_back(*(NeuronID*)(&(ac->at(s+count[i-1]*k))));
				}
			}
		}
	}
}


void SyncBuffer::null_terminate_send_buffer()
{
	// puts a "delta spike" just behind the last unrolled delay of the last group
	// std::cout << " term " << carry_offset << std::endl;
	send_buf.push_back(carry_offset);
}

void SyncBuffer::pop(SpikeDelay * delay, const NeuronID size)
{
	// TODO consider passing the current rank, because in principle it should not require the sync

	// clear all receiving buffers in the relevant time range
	for (NeuronID i = 1 ; i < MINDELAY+1 ; ++i ) {
		delay->get_spikes(i)->clear();
		delay->get_attributes(i)->clear();
	}

	// loop over different rank input segments in recv_buf
	for (int r = 0 ; r < mpicom->size() ; ++r ) {
		// reset time slice spike counts to extract correct number of attributes later
		for ( int i = 0 ; i < MINDELAY ; ++i ) count[i] = 0;

		//read current difference element from buffer and interpret as unrolled
		NeuronID * iter = &recv_buf[r*max_send_size+pop_offsets[r]]; // first spike

		AurynLong unrolled_spike = *iter;

		// handle overflow packages if there are any
		// TODO test overflow mechanism
		while ( *iter == std::numeric_limits<NeuronID>::max() ) {
			iter++; 
			unrolled_spike += *iter;
		}
	
		// subtract carry offset
		unrolled_spike -= pop_carry_offsets[r];

		// std::cout << "iter " << *iter << " unrolled " << unrolled_spike << std::endl;

		if ( unrolled_spike >= MINDELAY*size ) { // spike falls beyond all time slices of this group
			// increase carry by group size and carry on
			pop_carry_offsets[r] += MINDELAY*size;
			continue;
		}

		while ( unrolled_spike < MINDELAY*size ) { // one or more spikes belong to current group

			// decode spike positon on time slice grid
			const int slice = unrolled_spike/size;
			const NeuronID spike = unrolled_spike%size;
			
			// std::cout << "pop r:" << r << " slice: " << slice << " spike:" << spike << std::endl;

			// push spike to appropriate time slice
			delay->get_spikes(slice+1)->push_back(spike); 

			// store spike counts for each time-slice to decode the spike arguments correctly
			count[slice]++; 

			// set carry bit in case this spike puts us beyond the end and we leave the loop
			pop_carry_offsets[r] = MINDELAY*size-unrolled_spike;

			// advance iterator
			iter++; 
			unrolled_spike += *iter; // because we stored differences we need to add 
		}

		// extract a total of count*get_num_attributes() attributes 
		if ( delay->get_num_attributes() ) {
			for ( NeuronID slice = 0 ; slice < MINDELAY ; ++slice ) {
				AttributeContainer * ac = delay->get_attributes(slice+1);
				for ( int k = 0 ; k < delay->get_num_attributes() ; ++k ) { // loop over attributes
					for ( NeuronID s = 0 ; s < count[slice] ; ++s ) { // loop over spikes
						AurynFloat * attrib;
						attrib = (AurynFloat*)(iter);
						iter++;
						ac->push_back(*attrib);
// #ifdef DEBUG
// 						if ( mpicom->rank() == 0 )
// 							std::cout << " reading attr " 
// 								<< " " << slice << " "  
// 								<< k << " " << s << std::setprecision(5)
// 								<< " "  << *attrib << std::endl;
// #endif // DEBUG
					}
				}
			}
		}
		pop_offsets[r] = iter - &recv_buf[r*max_send_size]; // save offset in recv_buf section
	}

	// for( int i = 0; i < mpicom->size(); ++i ) {
	// 	MPI_Barrier( *mpicom );
	// 	if ( i == mpicom->rank() ) {
	// 		std::cout << "Rank " << mpicom->rank() << "pop " << "\n";
	// 		delay->print();
	// 	}
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

		if ( max_send_size > upper_estimate && max_send_size > 2 ) { 
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
		// std::cout << " sb size " << send_buf.size() << std::endl;
		ierr = MPI_Allgather(send_buf.data(), send_buf.size(), MPI_UNSIGNED, 
				recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
	} else { 
		// Create an overflow package 
		NeuronID * overflow_data  = new NeuronID[2];
		overflow_data[0] = -1;
		overflow_data[1] = send_buf.size(); 
		ierr = MPI_Allgather(overflow_data, 2, MPI_UNSIGNED, 
				recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
		delete overflow_data;
	}

	// error handling
	if ( ierr ) {
		std::cerr << "Error during MPI_Allgather." << std::endl;
		// TODO add an exceptoin to actually break the run here
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
		max_send_size = new_send_size+1;
		recv_buf.resize(mpicom->size()*max_send_size);
		// sync(); // recursive retry was causing problems
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
		pop_carry_offsets[i] = 0;
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
