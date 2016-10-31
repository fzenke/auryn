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

#include "DuplexConnection.h"

using namespace auryn;

void DuplexConnection::init() 
{
	if ( dst->get_post_size() == 0 ) {
		logger->debug("DuplexConnection:: Skipping init of bkw backward (transposed) matrix because post size is zero.");
		return; // TODO Test this recent addition of init guard
	}
	logger->debug("DuplexConnection:: Init of bkw backward (transposed) matrix");
	fwd = w; // for consistency declared here. fwd can be overwritten later though
	bkw = new BackwardMatrix ( get_n_cols(), get_m_rows(), w->get_nonzero() );
	allocated_bkw=true;
	compute_reverse_matrix();
}

void DuplexConnection::finalize() // finalize at this level is called only for reconnecting or non-Constructor building of the matrix
{
	std::stringstream oss;
	oss << "DuplexConnection: Finalizing ...";
	auryn::logger->msg(oss.str(),VERBOSE);

	compute_reverse_matrix();
}


DuplexConnection::DuplexConnection(const char * filename) 
: SparseConnection(filename)
{
	allocated_bkw=false;
	if ( dst->get_post_size() > 0 ) {
		init();
	} 
}

DuplexConnection::DuplexConnection(SpikingGroup * source, 
		NeuronGroup * destination, 
		TransmitterType transmitter) 
: SparseConnection(source, destination, transmitter)
{
	allocated_bkw=false;
}

DuplexConnection::DuplexConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename , 
		TransmitterType transmitter) 
: SparseConnection(source, destination, filename, transmitter)
{
	allocated_bkw=false;
	if ( dst->get_post_size() > 0 ) 
		init();
}


DuplexConnection::DuplexConnection(NeuronID rows, NeuronID cols) 
: SparseConnection(rows,cols)
{
	allocated_bkw=false;
	init();
}

DuplexConnection::DuplexConnection( SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		TransmitterType transmitter, std::string name) 
: SparseConnection(source,destination,weight,sparseness,transmitter, name)
{
	if ( dst->get_post_size() > 0 ) 
		init();
}

void DuplexConnection::free()
{
	logger->debug("DuplexConnection:: Freeing bkw backward/reverse/transposed matrix");
	delete bkw;
}


DuplexConnection::~DuplexConnection()
{
	if ( dst->get_post_size() != 0 && allocated_bkw ) free();
	else {  
		logger->debug("DuplexConnection:: Nothing to free.");
	}
}


void DuplexConnection::compute_reverse_matrix( int z )
{

	if ( fwd->get_nonzero() <= bkw->get_datasize() ) {
		auryn::logger->msg("Clearing reverse matrix..." ,VERBOSE);
		bkw->clear();
	} else {
		auryn::logger->msg("Bkw buffer too small reallocating..." ,VERBOSE);
		bkw->resize_buffer_and_clear(fwd->get_datasize());
	}

	std::stringstream oss;
	oss << "DuplexConnection: ("<< get_name() << "): Computing transposed (reverse) matrix view ...";
	auryn::logger->msg(oss.str(),VERBOSE);

	NeuronID maxrows = get_m_rows();
	NeuronID maxcols = get_n_cols();
	NeuronID ** rowwalker = new NeuronID * [maxrows+1];
	// copy pointer arrays
	for ( NeuronID i = 0 ; i < maxrows+1 ; ++i ) {
		rowwalker[i] = (fwd->get_rowptrs())[i];
	}
	for ( NeuronID j = 0 ; j < maxcols ; ++j ) {
		if ( dst->localrank(j) ) {
			for ( NeuronID i = 0 ; i < maxrows ; ++i ) {
				if (rowwalker[i] < fwd->get_rowptrs()[i+1]) { // stop when reached end of row
					if (*rowwalker[i]==j) { // if there is an element for that column add pointer to backward matrix
						bkw->push_back(j,i,fwd->get_ptr(i,j,z));
						++rowwalker[i];  // move on when processed element
					}
				}
			}
		}
	}
	bkw->fill_zeros();
	delete [] rowwalker;

	if ( fwd->get_nonzero() != bkw->get_nonzero() ) {
		oss.str("");
		oss << "DuplexConnection: ("<< get_name() << "): " 
			<< bkw->get_nonzero() 
			<< " different number of non-zero elements in bkw and fwd matrix.";
		auryn::logger->msg(oss.str(),ERROR);
	} else {
		oss.str("");
		oss << "DuplexConnection: ("<< get_name() << "): " 
			<< bkw->get_nonzero() 
			<< " elements processed.";
		auryn::logger->msg(oss.str(),VERBOSE);
	}
}

void DuplexConnection::prune(  )
{
	auryn::logger->msg("Pruning weight matrix",VERBOSE);
	fwd->prune();
	compute_reverse_matrix();
}

