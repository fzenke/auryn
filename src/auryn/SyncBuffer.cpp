/* 
* Copyright 2014-2023 Friedemann Zenke
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

	delete [] rank_send_sum;
	delete [] rank_send_sum2;
	delete [] rank_recv_count;
	delete [] rank_displs;
}

void SyncBuffer::init()
{

	/* Overflow value for syncbuffer */
	overflow_value = std::numeric_limits<NeuronID>::max();

	overflow_counter = 0;

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


	sync_counter = 0;
	max_send_size = 4;

	rank_send_sum = new int[mpicom->size()];
	rank_send_sum2 = new int[mpicom->size()];
	rank_recv_count = new int[mpicom->size()];
	rank_displs = new int[mpicom->size()];
	for ( int r = 0 ; r<mpicom->size() ; ++r ) {
		rank_send_sum[r] = 0;
		rank_send_sum2[r] = 0;
		rank_recv_count[r] = 2; // make space for overflow
	}

	resize_buffers(max_send_size);

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

int SyncBuffer::compute_buffer_margin(int n, int sum, int sum2)
{
	const int mean =  sum/n; 
	const int var  =  (sum2/n-mean*mean);
	const int marg =  SYNCBUFFER_SIZE_MARGIN_MULTIPLIER*std::sqrt(var);
	return marg;
}

int SyncBuffer::compute_buffer_size_with_margin(int n, int sum, int sum2)
{
	const int mean =  sum/n; 
	const int uest =  mean+compute_buffer_margin(n, sum, sum2);
	return uest;
}


void SyncBuffer::update_send_recv_counts()
{
	// update per rank send size estimates 
	for ( int i = 0 ; i<mpicom->size() ; ++i ) {
		const int uest =  compute_buffer_size_with_margin(sync_counter,rank_send_sum[i],rank_send_sum2[i]);
		if ( rank_recv_count[i] > uest && rank_recv_count[i] > 4 ) { 
			rank_recv_count[i] = rank_recv_count[i]+(uest-rank_recv_count[i])/2;
		} else {
			rank_recv_count[i] = uest;
		}

		rank_send_sum[i]  = 0;
		rank_send_sum2[i] = 0;
	}

	// find max send/recv value for max_send_size
	int new_max_send_size = 0;
	for ( int i = 0 ; i<mpicom->size() ; ++i ) {
		new_max_send_size = std::max(rank_recv_count[i],new_max_send_size);
	}

	if (max_send_size!=new_max_send_size) {
		max_send_size = new_max_send_size;
		resize_buffers(max_send_size);
	}

	sync_counter = 0;
}


void SyncBuffer::sync_allgatherv() 
{
	if ( sync_counter >= SYNCBUFFER_SIZE_HIST_LEN ) {  // update the estimate of maximum send size
		update_send_recv_counts();
	}

	int ierr = 0;

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	double T1, T2;              
    T1 = MPI_Wtime();     /* start time */
#endif

	if ( send_buf.size() <= rank_recv_count[mpicom->rank()] ) {
		send_buf[0] = send_buf.size();
		ierr = MPI_Allgatherv(send_buf.data(), rank_recv_count[mpicom->rank()], MPI_UNSIGNED,  
						      recv_buf.data(), rank_recv_count, rank_displs, MPI_UNSIGNED, *mpicom);
	} else { 
		// Create an overflow package 
		NeuronID * overflow_data;
		overflow_data = new NeuronID[rank_recv_count[mpicom->rank()]]; 
		// overflow_data = new NeuronID[max_send_size]; 
		overflow_data[0] = overflow_value;
		overflow_data[1] = send_buf.size(); 
		ierr = MPI_Allgatherv(overflow_data, rank_recv_count[mpicom->rank()], MPI_UNSIGNED,  
							  recv_buf.data(), rank_recv_count, rank_displs, MPI_UNSIGNED, *mpicom);
		delete [] overflow_data;
	}

	// error handling
	if ( ierr ) {
		std::cerr << "Error during MPI_Allgatherv." << std::endl;
		switch (ierr) {
			case MPI_ERR_COMM: std::cerr << "(MPI_ERR_COMM)." ; break;
			case MPI_ERR_COUNT: std::cerr << "(MPI_ERR_COUNT)." ; break;
			case MPI_ERR_TYPE: std::cerr << "(MPI_ERR_TYPE)." ; break;
			case MPI_ERR_BUFFER: std::cerr << "(MPI_ERR_BUFFER)." ; break;
			default: std::cerr << "ierr = " << ierr ; break;
		}
		std::cerr << std::endl;
		MPI::COMM_WORLD.Abort(-1);
	}

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
    T2 = MPI_Wtime();     /* end time */
	deltaT += (T2-T1);
#endif

	/* Detect overflow */
	bool overflow = false;
	int new_send_size = 0;
	for (int r = 0 ; r < mpicom->size() ; ++r ) {
		if  ( recv_buf[r*max_send_size]==overflow_value ) {
			overflow = true;
			const NeuronID value = recv_buf[r*max_send_size+1];
			rank_recv_count[r] = std::max(value,(unsigned int)2); // leave enough space for an overflow package
			if ( value > new_send_size ) {
				new_send_size = value;
			}
		}
	}


	if ( overflow ) {
		// std::cout << "overflow" << std::endl;
#ifdef DEBUG
		std::cerr << "Overflow in SyncBuffer adapting buffersize to "
			<< (new_send_size+2)*sizeof(NeuronID)
			<< " ( "
			<< mpicom->size()*(new_send_size+2)*sizeof(NeuronID)
			<< " total ) " 
			<< std::endl;
#endif //DEBUG
		++overflow_counter;
		max_send_size = std::max(new_send_size+2,max_send_size);
		resize_buffers(max_send_size);
		// resend full buffer
		ierr = MPI_Allgatherv(send_buf.data(), rank_recv_count[mpicom->rank()], MPI_UNSIGNED,  
							  recv_buf.data(), rank_recv_count, rank_displs, MPI_UNSIGNED, *mpicom);
	} 

	for ( int r = 0 ; r < mpicom->size() ; ++r ) { 
		const unsigned int val = recv_buf[r*max_send_size];
		rank_send_sum[r]  += val;
		rank_send_sum2[r] += val*val;
	}
	sync_counter++;

	// update senc/recv counters and buffers earlier if there was an overflow and we have some stats
	// if ( overflow && sync_counter>10 ) update_send_recv_counts(); 

	reset_send_buffer();
}


void SyncBuffer::sync_allgather() 
{
	if ( sync_counter >= SYNCBUFFER_SIZE_HIST_LEN ) {  // update the estimate of maximum send size
		update_send_recv_counts();
	}

	int ierr = 0;

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	double T1, T2;              
    T1 = MPI_Wtime();     /* start time */
#endif

	// if ( send_buf.size() <= rank_recv_count[mpicom->rank()] ) {
	if ( send_buf.size() <= max_send_size ) {
		send_buf[0] = send_buf.size();
		ierr = MPI_Allgather(send_buf.data(), max_send_size, MPI_UNSIGNED, 
							 recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
	} else { 
		// Create an overflow package 
		NeuronID * overflow_data;
		overflow_data = new NeuronID[max_send_size]; 
		overflow_data[0] = overflow_value;
		overflow_data[1] = send_buf.size(); 
		ierr = MPI_Allgather(overflow_data, max_send_size, MPI_UNSIGNED, 
							 recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
		delete [] overflow_data;
	}

	// error handling
	if ( ierr ) {
		std::cerr << "Error during MPI_Allgather." << std::endl;
		switch (ierr) {
			case MPI_ERR_COMM: std::cerr << "(MPI_ERR_COMM)." ; break;
			case MPI_ERR_COUNT: std::cerr << "(MPI_ERR_COUNT)." ; break;
			case MPI_ERR_TYPE: std::cerr << "(MPI_ERR_TYPE)." ; break;
			case MPI_ERR_BUFFER: std::cerr << "(MPI_ERR_BUFFER)." ; break;
			default: std::cerr << "ierr = " << ierr ; break;
		}
		std::cerr << std::endl;
		MPI::COMM_WORLD.Abort(-1);
	}

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
    T2 = MPI_Wtime();     /* end time */
	deltaT += (T2-T1);
#endif

	/* Detect overflow */
	bool overflow = false;
	int new_send_size = 0;
	for (int r = 0 ; r < mpicom->size() ; ++r ) {
		if  ( recv_buf[r*max_send_size]==overflow_value ) {
			overflow = true;
			const NeuronID value = recv_buf[r*max_send_size+1];
			rank_recv_count[r] = std::max(value,(unsigned int)2); // leave enough space for an overflow package
			if ( value > new_send_size ) {
				new_send_size = value;
			}
		}
	}


	if ( overflow ) {
#ifdef DEBUG
		std::cerr << "Overflow in SyncBuffer adapting buffersize to "
			<< (new_send_size+2)*sizeof(NeuronID)
			<< " ( "
			<< mpicom->size()*(new_send_size+2)*sizeof(NeuronID)
			<< " total ) " 
			<< std::endl;
#endif //DEBUG
		++overflow_counter;
		max_send_size = std::max(new_send_size+2,max_send_size);
		resize_buffers(max_send_size);
		// resend full buffer
		ierr = MPI_Allgather(send_buf.data(), max_send_size, MPI_UNSIGNED, 
							 recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
	} 

	for ( int r = 0 ; r < mpicom->size() ; ++r ) { 
		const unsigned int val = recv_buf[r*max_send_size];
		rank_send_sum[r]  += val;
		rank_send_sum2[r] += val*val;
	}
	sync_counter++;

	// update senc/recv counters and buffers earlier if there was an overflow and we have some stats
	// if ( overflow && sync_counter>10 ) update_send_recv_counts(); 

	reset_send_buffer();
}

void SyncBuffer::sync() 
{
	// allgather seems to be faster on the standard sim_background benchmark
	// however, replacing the following line by synchronize will run the new
	// code using Allgatherv in which each rank can send different amounts of data.
	sync_allgather();
}

void SyncBuffer::reset_send_buffer()
{
	send_buf.clear();
	send_buf.push_back(0);

	// reset carry offsets for push and pop functions
	carry_offset = 0;
	for ( int i = 0 ; i < mpicom->size() ; ++i ) { 
		pop_offsets[i] = 1;
		pop_delta_spikes[i] = undefined_delta_size;
	}
}

void SyncBuffer::resize_buffers(NeuronID send_size)
{
	send_buf.reserve(send_size);
	recv_buf.resize(mpicom->size()*send_size);
	// std::cout << "cap " << recv_buf.capacity() << std::endl;
	for ( int r = 0 ; r<mpicom->size() ; ++r ) {
		rank_displs[r] = r*send_size;
	}
}

int SyncBuffer::get_max_send_buffer_size()
{
	return max_send_size;
}

unsigned int SyncBuffer::get_overflow_count()
{
	return overflow_counter;
}

unsigned int SyncBuffer::get_sync_count()
{
	return sync_counter;
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
