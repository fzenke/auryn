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

#include "SyncBuffer.h"

SyncBuffer::SyncBuffer( mpi::communicator * com )
{
	mpicom = com;
	init();
}

void SyncBuffer::init()
{
	// for ( NeuronID i = 0 ; i < SYNCBUFFER_SIZE_HIST_LEN ; ++i )
	// 	size_history[i] = 0;
	// size_history_ptr = 0;
	

	for ( NeuronID i = 0 ; i < mpicom->size() ; ++i )
		pop_offsets.push_back(1);

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

    // add a size check here
	// sizeof(NeuronID) 
	// sizeof(AurynFloat);
}

void SyncBuffer::push(SpikeDelay * delay, NeuronID size)
{

	for (NeuronID i = 1 ; i < MINDELAY+1 ; ++i ) {
		SpikeContainer * sc = delay->get_spikes(i);

		// NeuronID s = (NeuronID) (sc->size());
		// send_buf[0] += s;

		count[i-1] = 0;
		for (SpikeContainer::const_iterator spike = sc->begin() ; 
			spike != sc->end() ; ++spike ) {
			NeuronID compressed = *spike + groupPushOffset1 + (i-1)*size;
			send_buf.push_back(compressed);
			count[i-1]++;
		}

		send_buf[0] += delay->get_spikes(i)->size(); // send the total number of spikes
	}

	// transmit get_num_attributes() attributes for count spikes for all time slices
	if ( delay->get_num_attributes() ) {
		for (NeuronID i = 1 ; i < MINDELAY+1 ; ++i ) {
			AttributeContainer * ac = delay->get_attributes(i);
			for ( NeuronID k = 0 ; k < delay->get_num_attributes() ; ++k ) { // loop over attributes
				for ( NeuronID s = 0 ; s < count[i-1] ; ++s ) { // loop over spikes
					send_buf.push_back(*(NeuronID*)(&(ac->at(s+count[i-1]*k))));
					// if ( mpicom->rank() == 0 )
					// 	cout << " pushing attr " << " " << i << " " << k << " " << s << " " 
					// 		<< scientific << ac->at(s+count[i-1]*k) << endl;
				}
			}
		}
	}

	groupPushOffset1 += size*MINDELAY;
}

void SyncBuffer::pop(SpikeDelay * delay, NeuronID size)
{
	for (NeuronID i = 1 ; i < MINDELAY+1 ; ++i ) {
		delay->get_spikes(i)->clear();
		delay->get_spikes(i)->clear();
	}


	for (int r = 0 ; r < mpicom->size() ; ++r ) {
		NeuronID numberOfSpikes = recv_buf[r*max_send_size]; // total data
		NeuronID * iter = &recv_buf[r*max_send_size+pop_offsets[r]]; // first spike

		NeuronID temp  = (*iter - groupPopOffset);
		NeuronID spike = temp%size; // spike (if it exists) in current group
		int t = temp/size; // timeslice in MINDELAY if we are out this might be the next group


		for ( int i = 0 ; i < MINDELAY ; ++i ) count[i] = 0;
		// while we are in the current group && have not read all entries
		while ( t < MINDELAY && numberOfSpikes ) {  
			delay->get_spikes(t+1)->push_back(spike);
			iter++;
			numberOfSpikes--;
			count[t]++; // store spike counts for each time-slice

			temp  = (*iter - groupPopOffset);
			spike = temp%size;
			t = temp/size;
		}


		// extract a total of count*get_num_attributes() attributes 
		if ( delay->get_num_attributes() ) {
			for ( NeuronID slice = 0 ; slice < MINDELAY ; ++slice ) {
				AttributeContainer * ac = delay->get_attributes(slice+1);
				for ( NeuronID k = 0 ; k < delay->get_num_attributes() ; ++k ) { // loop over attributes
					for ( NeuronID s = 0 ; s < count[slice] ; ++s ) { // loop over spikes
						AurynFloat * attrib;
						attrib = (AurynFloat*)(iter);
						iter++;
						ac->push_back(*attrib);
						// if ( mpicom->rank() == 0 )
						// 	cout << " reading attr " << " " << slice << " "  << k << " " << s << " " << scientific << *attrib << endl;
					}
				}
			}
		}

		recv_buf[r*max_send_size] = numberOfSpikes; // save remaining entries
		pop_offsets[r] = iter - &recv_buf[r*max_send_size]; // save offset in recv_buf section
	}

	groupPopOffset += size*MINDELAY;

}


void SyncBuffer::sync() 
{
	if ( syncCount >= SYNCBUFFER_SIZE_HIST_LEN ) {  // update the estimate of maximum send size
		NeuronID mean_send_size =  maxSendSum/syncCount; // allow for 5 times the max mean
		NeuronID var_send_size  =  (maxSendSum2-mean_send_size*mean_send_size)/syncCount;
		NeuronID upper_estimate =  mean_send_size+SYNCBUFFER_SIZE_MARGIN_MULTIPLIER*sqrt(var_send_size);

		if ( max_send_size > upper_estimate && max_send_size > 2 ) { 
			max_send_size = (max_send_size+upper_estimate)/2;
			recv_buf.resize(mpicom->size()*max_send_size);
		}	
		maxSendSum = 0;
		maxSendSum2 = 0;
		syncCount = 0;
	}

	int ierr;

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	double T1, T2;              
    T1 = MPI_Wtime();     /* start time */
#endif
	if ( send_buf.size() <= max_send_size ) {
		ierr = MPI_Allgather(send_buf.data(), send_buf.size(), MPI_UNSIGNED, 
				recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
	} else { 
		// Create a overflow package 
		NeuronID * overflow_data  = new NeuronID[2];
		overflow_data[0] = -1;
		overflow_data[1] = send_buf.size(); 
		ierr = MPI_Allgather(overflow_data, 2, MPI_UNSIGNED, 
				recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
		delete overflow_data;
	}

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
    T2 = MPI_Wtime();     /* end time */
	deltaT += (T2-T1);
#endif

	/* Detect over flow */
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
		// cout << "Overflow in SyncBuffer adapting buffersize to "
		// 	<< (new_send_size+1)*sizeof(NeuronID)
		// 	<< " ( "
		// 	<< mpicom->size()*(new_send_size+1)*sizeof(NeuronID)
		// 	<< " total ) " 
		// 	<< endl;
		max_send_size = new_send_size+1;
		recv_buf.resize(mpicom->size()*max_send_size);
		// sync(); // recursive retry was ausing problems
		// resend full buffer
		ierr = MPI_Allgather(send_buf.data(), send_buf.size(), MPI_UNSIGNED, 
				recv_buf.data(), max_send_size, MPI_UNSIGNED, *mpicom);
	} 

	// reset
	NeuronID largest_message = 0;
	for (vector<NeuronID>::iterator iter = pop_offsets.begin() ;
			iter != pop_offsets.end() ;
			++iter ) {
		largest_message = max(*iter,largest_message);
		*iter = 1;
	}
	maxSendSum += largest_message;
	maxSendSum2 += largest_message*largest_message;


	syncCount++;

	reset_send_buffer();
}

void SyncBuffer::reset_send_buffer()
{
	send_buf.clear();
	send_buf.push_back(0); // initial size first entry
	groupPushOffset1 = 0;
	groupPopOffset = 0;
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
