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

#include "SparseConnection.h"

// static members
boost::mt19937 SparseConnection::sparse_connection_gen = boost::mt19937();
bool SparseConnection::has_been_seeded = false;


SparseConnection::SparseConnection(const char * filename) : Connection()
{
	init();
	if (! init_from_file(filename) )
		throw AurynMMFileException();
}

SparseConnection::SparseConnection(SpikingGroup * source, NeuronGroup * destination, const char * filename, TransmitterType transmitter) : Connection(source, destination, transmitter)
{
	init();
	if (! init_from_file(filename) )
		throw AurynMMFileException();
}

SparseConnection::SparseConnection(NeuronID rows, NeuronID cols) : Connection(rows,cols)
{
	init();
}

SparseConnection::SparseConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) : Connection(source,destination,transmitter)
{
	init();
}


SparseConnection::SparseConnection( 
		SpikingGroup * source, 
		NeuronGroup * destination, 
		AurynWeight weight, 
		AurynFloat sparseness, 
		TransmitterType transmitter, 
		string name) 
	: Connection(source,destination,transmitter,name)
{
	init();
	stringstream oss;
	AurynLong anticipatedsize = (AurynLong) (estimate_required_nonzero_entires ( sparseness*src->get_pre_size()*dst->get_post_size() ) );
	oss << "SparseConnection: ("<< get_name() <<"): Assuming memory demand for pre #" << src->get_pre_size() << " and post #" << dst->get_post_size() 
													<< std::scientific << setprecision(4) << " ( total " << anticipatedsize << ")";
	logger->msg(oss.str(),DEBUG);
	allocate(anticipatedsize);
	connect_random(weight,sparseness,skip_diagonal);
}


SparseConnection::SparseConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, 
		AurynFloat sparseness, 
		NeuronID lo_row, 
		NeuronID hi_row, 
		NeuronID lo_col, 
		NeuronID hi_col, 
		TransmitterType transmitter) 
	: Connection(source,destination,transmitter)
{
	NeuronID rows = hi_row - lo_row;
	NeuronID cols = hi_col - lo_col;
	init();
	allocate((AurynLong) ( estimate_required_nonzero_entires( sparseness*rows*cols ) ));
	w->clear();
	connect_block_random(weight,sparseness,lo_row,hi_row,lo_col,hi_col,skip_diagonal);
	finalize();
}


SparseConnection::SparseConnection(
		SpikingGroup * source, 
		NeuronGroup * destination, 
		SparseConnection * con, 
		string name ) 
	: Connection(source,destination)
{
	set_transmitter(con->get_transmitter());
	double sparseness = get_nonzero()/(con->get_m_rows()*con->get_n_cols());
	AurynFloat mean,std; con->stats(mean,std);
	AurynLong anticipatedsize = (AurynLong) (estimate_required_nonzero_entires ( sparseness*src->get_pre_size()*dst->get_post_size() ) );
	allocate(anticipatedsize);
	connect_random(mean,sparseness,con->skip_diagonal);
}

SparseConnection::~SparseConnection()
{
	free();
}

void SparseConnection::init() 
{
	if ( dst->evolve_locally() == true )
		sys->register_connection(this);
	has_been_allocated = false;
	if ( src == dst ) {
		skip_diagonal = true; 
		stringstream oss;
		oss << "SparseConnection: ("<< get_name() <<"): Detected recurrent connection. skip_diagonal was activated!";
		logger->msg(oss.str(),DEBUG);
	}
	else skip_diagonal = false;

	if ( !has_been_seeded ) { // seed it only once 
		int rseed = 12345*communicator->rank() ;
		seed(rseed);
	}

	set_min_weight(0.0);
	set_max_weight(1e16); // just make it large 

	patterns_ignore_gamma = false;
	wrap_patterns = false;

	patterns_every_pre = 1;
	patterns_every_post = 1;
}

void SparseConnection::seed(NeuronID randomseed) 
{
	stringstream oss;
	oss << "SparseConnection: ("<< get_name() <<"): Seeding with " << randomseed;
	logger->msg(oss.str(),DEBUG);
	SparseConnection::sparse_connection_gen.seed(randomseed); 
	has_been_seeded = true;
}

AurynLong SparseConnection::estimate_required_nonzero_entires( AurynLong nonzero , double sigma )
{
	return nonzero + sigma*sqrt(nonzero) ;
}

void SparseConnection::free()
{
	delete w;
}


void SparseConnection::allocate(AurynLong bufsize)
{
	NeuronID m = get_m_rows();  
	NeuronID n = get_n_cols();

	stringstream oss;
	oss << "SparseConnection: (" << get_name() << "): Allocating sparse matrix (" << m << ", " << n << ") with space for "  << std::scientific << setprecision(4) << (double) bufsize <<  " nonzero elements ";
	logger->msg(oss.str(),DEBUG);

	AurynLong maxsize = m*n;
	
	w = new ForwardMatrix ( m, n , min(maxsize,bufsize) );

	has_been_allocated = true;
}

void SparseConnection::allocate_manually(AurynLong expected_size)
{
	AurynDouble dilution = (AurynDouble) dst->get_rank_size()/dst->get_size();
	allocate( estimate_required_nonzero_entires ( dilution*expected_size ) );
}

void SparseConnection::set_min_weight(AurynWeight minimum_weight)
{
	wmin = minimum_weight;
}

AurynWeight SparseConnection::get_min_weight()
{
	return wmin;
}

void SparseConnection::set_max_weight(AurynWeight maximum_weight)
{
	wmax = maximum_weight;
}

AurynWeight SparseConnection::get_max_weight()
{
	return wmax;
}

void SparseConnection::random_data(AurynWeight mean, AurynWeight sigma) 
{
	stringstream oss;
	oss << "SparseConnection: (" << get_name() << "): randomizing non-zero connections (gaussian) with mean=" << mean << " sigma=" << sigma ;
	logger->msg(oss.str(),NOTIFICATION);

	boost::normal_distribution<> dist((double)mean, (double)sigma);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(SparseConnection::sparse_connection_gen, dist);
	AurynWeight rv;

	for ( AurynLong i = 0 ; i<w->get_nonzero() ; ++i ) {
		rv = die();
		if ( rv<get_min_weight() ) rv = get_min_weight();
		w->set_data(i,rv);
	}
}

void SparseConnection::sparse_set_data(AurynDouble sparseness, AurynWeight value) 
{
	stringstream oss;
	oss << "SparseConnection: (" << get_name() << "): setting data sparsely with sparseness=" << sparseness << " value=" << value ;
	logger->msg(oss.str(),DEBUG);

	boost::exponential_distribution<> dist(sparseness);
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(SparseConnection::sparse_connection_gen, dist);

    AurynLong x = (AurynLong) die();
    AurynLong stop = w->get_datasize();

	while ( x < stop ) {
		set_data(x,value);
		x += die();
	}
}

void SparseConnection::random_col_data(AurynWeight mean, AurynWeight sigma) 
{
	stringstream oss;
	oss << "SparseConnection: (" << get_name() << "): Randomly scaling cols (gaussian) with mean=" << mean << " sigma=" << sigma ;
	logger->msg(oss.str(),DEBUG);

	boost::normal_distribution<> dist((double)mean, (double)sigma);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(SparseConnection::sparse_connection_gen, dist);
	AurynWeight rv;

	for ( NeuronID i = 0 ; i<get_n_cols() ; ++i ) {
		rv = die();
		if ( rv<0 ) rv = 0;
		w->set_col(i,rv);
	}
}


void SparseConnection::set_block(NeuronID lo_row, NeuronID hi_row, NeuronID lo_col, NeuronID hi_col, AurynWeight weight)
{
	AurynWeight temp = max(weight,get_min_weight());
	for ( NeuronID i = 0 ; i < get_m_rows() ; ++i ) 
	{
		for ( NeuronID * j = w->get_row_begin(i) ; j != w->get_row_end(i) ; ++j )
		{
			if (i >= lo_row && i < hi_row && *j >= lo_col && *j < hi_col )
			  w->get_data_begin()[j-w->get_row_begin(0)] = temp;
		}
	}
}

void SparseConnection::set_all(AurynWeight weight)
{
	w->set_all( weight );
}

void SparseConnection::scale_all(AurynFloat value)
{
	w->scale_all( value );
}

void SparseConnection::set_upper_triangular(AurynWeight weight)
{
	w->set_all( 0.0 );
	AurynWeight temp = max(weight,get_min_weight());
	for ( NeuronID i = 0 ; i < get_m_rows() ; ++i ) 
	{
		for ( NeuronID * j = w->get_row_begin(i) ; j != w->get_row_end(i) ; ++j )
		{
			if ( i <=  *j )
			  w->get_data_begin()[j-w->get_row_begin(0)] = temp;
		}
	}
}


void SparseConnection::connect_block_random(AurynWeight weight, float sparseness,
		NeuronID lo_row, 
		NeuronID hi_row, 
		NeuronID lo_col, 
		NeuronID hi_col, 
		bool skip_diag )
{
	int r = 0; // these variables are used to speed up building the matrix if the destination is distributed
	int s = 1;

	r = communicator->rank()-dst->get_locked_rank(); 
	s = dst->get_locked_range();

	boost::exponential_distribution<> dist(sparseness);
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(SparseConnection::sparse_connection_gen, dist);

	if (!has_been_allocated)
		throw AurynConnectionAllocationException();

	AurynLong idim = (hi_row-lo_row);
	AurynLong jdim = (hi_col-lo_col)/s;
	if ( (hi_col-lo_col)%s > r ) { // some ranks have one more "carry" neuron
		jdim += 1;
	}
    AurynLong x = (AurynLong) die();
	if ( sparseness == 1.0 ) x = 0; // for dense matrices
    AurynLong stop = idim*jdim;
    AurynLong count = 0;
    NeuronID i = 0;
    NeuronID j = 0;
    while ( x < stop ) {
		i = lo_row+x/jdim;
		j = lo_col + s*(x%jdim) + r; // be carfule with this line ... it already was the cause of a lot of headaches
		if ( (j >= lo_col) && (!skip_diag || i!=j)) {
			try {
				if ( push_back(i,j,weight) )
					count++;
			}
			catch ( AurynMatrixDimensionalityException )
			{
				stringstream oss;
				oss << "SparseConnection: ("
					<< get_name() 
					<<"): Trying to add elements outside of matrix (i=" 
					<< i 
					<< "j="
					<< j 
					<< ", "
					<< count 
					<< "th element) ";
				logger->msg(oss.str(),ERROR);
				return;
			} 
			catch ( AurynMatrixPushBackException )
			{
				stringstream oss;
				oss << "SparseConnection: ("<< get_name() 
					<< "): Failed pushing back element. Maybe due to out of order pushing? "
					<< " (" << i << "," << j << ") "
					<< " with count=" << count 
					<< " in connect_block_random ( fill_level= " << w->get_fill_level() << " )";
				logger->msg(oss.str(),ERROR);
				return;
			} 
			catch ( AurynMatrixBufferException )
			{
				stringstream oss;
				oss << "SparseConnection: ("
					<< get_name() 
					<<"): Buffer full after pushing " 
					<< count 
					<< " elements."
					<< " There are pruned connections!";
				logger->msg(oss.str(),ERROR);
				return;
			} 
		}
		AurynLong jump = (AurynLong) (die()+0.5);
		if ( jump == 0 || sparseness >= 1.0 )  
			x += 1 ;
		else
			x += jump ;
	}

	stringstream oss;
	oss << "SparseConnection: ("<< get_name() <<"): Finished connect_block_random ["
		<< lo_row << ", " << hi_row << ", " << lo_col << ", " << hi_col <<  "] " << " (stop count " 
		<< std::scientific << setprecision(4) << (double) stop 
		<< ") and successfully pushed " << (double) count <<  " entries. " 
    	<< "Resulting overall sparseness " << 1.*get_nonzero()/src->get_pre_size()/dst->get_post_size();
	logger->msg(oss.str(),DEBUG);
}

void SparseConnection::connect_random(AurynWeight weight, float sparseness, bool skip_diag)
{
	if ( dst->evolve_locally() ) { // if there are no local units there is no need for synapses
		stringstream oss;
		oss << "SparseConnection: ("<< get_name() <<"): Randomfill with weight "<< weight <<  " and sparseness " << sparseness;
		logger->msg(oss.str(),DEBUG);
		w->clear();
		connect_block_random(weight,sparseness,0,get_m_rows(),0,get_n_cols(),skip_diag);
	}
	finalize();
}

void SparseConnection::finalize()
{
	w->fill_zeros();
	if ( dst->evolve_locally() ) {
		stringstream oss;
		oss << "SparseConnection: ("<< get_name() <<"): Finalized with fill level " << w->get_fill_level();
		logger->msg(oss.str(),DEBUG);
		if (w->get_fill_level()<WARN_FILL_LEVEL)
		{
			stringstream oss2;
			oss2 << "SparseConnection: ("<< get_name() <<"): Wasteful fill level (" << w->get_fill_level() << ")! Make sure everything is in order!";
			logger->msg(oss2.str(),WARNING);
		}
	}
}

bool SparseConnection::push_back(NeuronID i, NeuronID j, AurynWeight weight) 
{
	if ( dst->localrank(j) ) {
		w->push_back(i,j,weight);
		return true;
	}
	return false;
}

void SparseConnection::propagate()
{
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ;
			spike != src->get_spikes()->end() ; 
			++spike ) {
		for ( AurynLong c = w->get_row_begin_index(*spike) ;
				c < w->get_row_end_index(*spike) ;
				++c ) {
			transmit( w->get_colind(c) , w->get_value(c) );
		}
	}
}

void SparseConnection::sanity_check()
{
	if ( dst->evolve_locally() == false ) return;

	AurynFloat * sum = new AurynFloat[dst->get_size()];
	for ( NeuronID i = 0 ; i < dst->get_size() ; ++i ) sum[i] = 0.0;

	NeuronID * ind = w->get_ind_begin(); // first element of index array
	AurynWeight * data = w->get_data_begin();
	for ( NeuronID i = 0 ; i < src->get_size() ; ++i ) {
		for (NeuronID * c = w->get_row_begin(i) ; 
				c < w->get_row_end(i) ; 
				++c ) {
			AurynWeight value = data[c-ind]; 
			sum[*c] += value;
		}
	}

	NeuronID unconnected_count = 0 ;
	double total_weight = 0;
	for ( NeuronID i = 0 ; i < dst->get_size() ; ++i ) {
		total_weight += sum[i];
		if ( sum[i] == 0 && dst->localrank(i) ) {
			unconnected_count++;
			stringstream oss;
			oss << "Sanity check: Neuron "
				<< i 
				<< " local (" 
				<< dst->global2rank(i) 
				<< ") has no inputs." ;
			logger->msg(oss.str(),WARNING);
		}
	}

	logger->parameter("sanity_check:total weight",total_weight);

	if ( unconnected_count ) { 
		stringstream oss;
		oss << "Sanity check failed ("
			<< get_name()
			<< "). Found " 
			<< unconnected_count
			<< " unconnected neurons.";
		logger->msg(oss.str(),WARNING);
	}

	delete [] sum;

	//  row count - outputs

	AurynFloat * sum_rows = new AurynFloat[src->get_size()];
	for ( NeuronID i = 0 ; i < src->get_size() ; ++i ) sum_rows[i] = 0.0;

	for ( NeuronID i = 0 ; i < src->get_size() ; ++i ) {
		for (NeuronID * c = w->get_row_begin(i) ; 
				c != w->get_row_end(i) ; 
				++c ) {
			AurynWeight value = data[c-ind]; 
			sum_rows[i] += value;
		}
	}

	for ( NeuronID i = 0 ; i < src->get_size() ; ++i ) {
		if ( sum_rows[i] == 0  ) {
			stringstream oss;
			oss << "Sanity check: Neuron "
			 << i 
			 << " local (" 
			 << src->global2rank(i) 
			 << ") has no outputs on this rank. This might be normal when run distributed." ;
			logger->msg(oss.str(),DEBUG);
			if ( w->get_row_begin(i) != w->get_row_end(i) ) {
				logger->msg("wmat inconsistency",ERROR);
			}
		}
	}


	delete [] sum_rows;
}

void SparseConnection::stats(AurynFloat &mean, AurynFloat &std)
{
	NeuronID count = 0;
	AurynFloat sum = 0;
	AurynFloat sum2 = 0;
	// for ( NeuronID i = 0 ; i < get_m_rows() ; ++i ) 
	// {
	// 	for ( NeuronID * j = w->get_row_begin(i) ; j != w->get_row_end(i) ; ++j )
	// 	{
	// 		count++;
	// 		t = w->get_data_begin()[j-w->get_row_begin(0)];
	// 		sum += t;
	// 		sum2 += (t*t);
	// 	}
	// }
	for ( AurynWeight * iter = w->get_data_begin() ; iter != w->get_data_end() ; ++iter ) {
		sum  += *iter;
		sum2 += (*iter * *iter);
	}
	count = w->get_nonzero();
	if ( count <= 1 ) {
		mean = sum;
		std = 0;
		return;
	}
	mean = sum/count;
	std = sqrt(sum2/count-mean*mean);
}

AurynDouble SparseConnection::sum()
{
	AurynFloat sum = 0;

	for ( AurynWeight * iter = w->get_data_begin() ; iter != w->get_data_end() ; ++iter ) {
		sum  += *iter;
	}
	
	return sum;
}

AurynWeight SparseConnection::get_data(NeuronID i)
{
	return w->get_data(i);
}

void SparseConnection::set_data(NeuronID i, AurynWeight value)
{
	w->set_data(i,value);
}

AurynWeight SparseConnection::get(NeuronID i, NeuronID j)
{
	return w->get(i,j);
}

AurynWeight * SparseConnection::get_ptr(NeuronID i, NeuronID j)
{
	return w->get_ptr(i,j);
}

void SparseConnection::set(vector<neuron_pair> element_list, AurynWeight value)
{
	for (vector<neuron_pair>::iterator iter = element_list.begin() ; iter != element_list.end() ; ++iter)
	{
		w->set((*iter).i, (*iter).j,value);
	}
}

void SparseConnection::set(NeuronID i, NeuronID j, AurynWeight value)
{
	value = max(value,get_min_weight());
	w->set(i,j,value);
}

bool SparseConnection::write_to_file(ForwardMatrix * m, string filename )
{
	if ( !dst->evolve_locally() ) return true;

	ofstream outfile;
	outfile.open(filename.c_str(),ios::out);
	if (!outfile) {
		stringstream oss;
	    oss << "Can't open output file " << filename;
		logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	outfile << "%%MatrixMarket matrix coordinate real general\n" 
		<< "% Auryn weight matrix. Has to be kept in row major order for load operation.\n" 
		<< "% Connection name: " << get_name() << "\n"
		<< "% Locked range: " << dst->get_locked_range() << "\n"
		<< "%\n"
		<< get_m_rows() << " " << get_n_cols() << " " << m->get_nonzero() << endl;

	for ( NeuronID i = 0 ; i < get_m_rows() ; ++i ) 
	{
		outfile << setprecision(7);
		for ( NeuronID * j = m->get_row_begin(i) ; j != m->get_row_end(i) ; ++j )
		{
			outfile << i+1 << " " << *j+1 << " " << scientific << m->get_data_begin()[j-m->get_row_begin(0)] << fixed << "\n";
		}
	}

	outfile.close();
	return true;
}

bool SparseConnection::write_to_file(string filename)
{
	return write_to_file(w,filename.c_str());
}

AurynLong SparseConnection::dryrun_from_file(string filename)
{
	if ( !dst->evolve_locally() ) return 0;

	char buffer[256];
	ifstream infile (filename.c_str());
	if (!infile) {
		stringstream oss;
		oss << "Can't open input file " << filename;
		logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	NeuronID i,j;
	AurynLong k;
	unsigned int count = 0;
	unsigned int pushback_count = 0;
	float val;

	// read connection details
	infile.getline (buffer,256); count++;
	string header("%%MatrixMarket matrix coordinate real general");
	if (header.compare(buffer)!=0)
	{
		stringstream oss;
		oss << "Input format not recognized.";
		logger->msg(oss.str(),ERROR);
		return false;
	}
	while ( buffer[0]=='%' ) {
	  infile.getline (buffer,256);
	  count++;
	}

	sscanf (buffer,"%u %u %lu",&i,&j,&k);

	while ( infile.getline (buffer,256) )
	{
		count++;
		sscanf (buffer,"%u %u %e",&i,&j,&val);
		if ( (i-1) < src->get_size() && dst->localrank(j-1) )
			pushback_count++;
	}

	infile.close();

	return pushback_count;
}

bool SparseConnection::load_from_file(ForwardMatrix * m, string filename, AurynLong data_size )
{
	if ( !dst->evolve_locally() ) return true;

	char buffer[256];
	ifstream infile (filename.c_str());
	if (!infile) {
		stringstream oss;
		oss << "Can't open input file " << filename;
		logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	set_name(filename);

	NeuronID i,j;
	AurynLong k;
	unsigned int count = 0;
	unsigned int pushback_count = 0;
	float val;

	// read connection details
	infile.getline (buffer,256); count++;
	string header("%%MatrixMarket matrix coordinate real general");
	if (header.compare(buffer)!=0)
	{
		stringstream oss;
		oss << "Input format not recognized.";
		logger->msg(oss.str(),ERROR);
		return false;
	}
	while ( buffer[0]=='%' ) {
	  infile.getline (buffer,256);
	  count++;
	}

	sscanf (buffer,"%u %u %lu",&i,&j,&k);
	set_size(i,j);

	if ( data_size ) { 
		k = data_size;
	}
	
	if ( m->get_datasize() >= k ) {
		m->clear();
	} else {
		stringstream oss;
		oss << "Buffer too small ("
			<< m->get_datasize()
			<< " -> "
			<< k
			<< " elements). Reallocating.";
		logger->msg(oss.str() ,NOTIFICATION);
		m->resize_buffer_and_clear(k);
	}

	stringstream oss;
	oss << get_name() 
		<< ": Reading from file ("
		<< get_m_rows()<<"x"<<get_n_cols()
		<< " @ "<<1.*k/(src->get_size()*dst->get_rank_size())<<")";
	logger->msg(oss.str(),NOTIFICATION);


	while ( infile.getline (buffer,255) )
	{
		count++;
		sscanf (buffer,"%u %u %e",&i,&j,&val);
		try {
			if ( dst->localrank(j-1) ) {
				m->push_back(i-1,j-1,val);
				pushback_count++;
			}
		}
		catch ( AurynMatrixPushBackException )
		{
			stringstream oss;
			oss << "Push back failed. Error in line=" << count << ", "
				<< " i=" << i 
				<< " j=" << j 
				<< " v=" << val << ". "
				<< " After pushing " << pushback_count << " elements. "
				<< " Bad row major order?";
			logger->msg(oss.str(),ERROR);
			throw AurynMMFileException();
			return false;
		} 
		catch ( AurynMatrixBufferException )
		{
			stringstream oss;
			oss << get_name() 
				<< ": Buffer full after pushing " 
				<< count << " elements."
				<< " There are pruned connections!";
			logger->msg(oss.str(),ERROR);
			return false;
		} 
		catch ( AurynMatrixDimensionalityException )
		{
			stringstream oss;
			oss << "SparseConnection: ("
				<< get_name() 
				<<"): Trying to add elements outside of matrix (i=" 
				<< i 
				<< ", j="
				<< j 
				<< ", "
				<< count 
				<< "th element) ";
			logger->msg(oss.str(),ERROR);
			return false;
		} 
	}


	infile.close();

	if ( pushback_count != m->get_nonzero() ) { // this should never happen without an exception above, but better be save than sorry
		oss.str("");
		oss << get_name() << ": " 
			<< pushback_count 
			<< " elements pushed, but only "
			<< m->get_nonzero()
			<< " in matrix matrix.";
		logger->msg(oss.str(),ERROR);
	} else {
		oss.str("");
		oss << get_name() << ": OK, " 
			<< pushback_count 
			<< " elements pushed.";
		logger->msg(oss.str(),DEBUG);
	}

	m->fill_zeros();
	// finalize(); // commented this line out because it only acts on w

	return true;
}

bool SparseConnection::load_from_complete_file(string filename)
{
	AurynLong datasize = dryrun_from_file(filename);
	stringstream oss;
	oss << "Loading from complete file. Element count: "
		<< datasize 
		<< ".";
	logger->msg(oss.str(),NOTIFICATION);
	bool returnvalue = load_from_file(w,filename,datasize);
	finalize();
	return returnvalue;
}

bool SparseConnection::load_from_file(string filename)
{
	bool result = load_from_file(w,filename);
	finalize();
	return result;
}

bool SparseConnection::init_from_file(const char * filename)
{
	allocate(1);
	return load_from_file(filename);
}


AurynLong SparseConnection::get_nonzero()
{
	return w->get_nonzero();
}

void SparseConnection::put_pattern( type_pattern * pattern, AurynWeight strength, bool overwrite )
{
	stringstream oss;
	oss << "SparseConnection: ("<< get_name() <<"): Putting assembly ( size " << pattern->size() << " )";
	logger->msg(oss.str(),DEBUG);

	put_pattern( pattern, pattern, strength, overwrite );
}

void SparseConnection::put_pattern( type_pattern * pattern1, type_pattern * pattern2, AurynWeight strength, bool overwrite )
{
	type_pattern::iterator iter_pre,iter_post;
	for ( iter_post = pattern2->begin() ; iter_post != pattern2->end() ; ++iter_post ) {
		if ( (*iter_post).i%patterns_every_post != 0 ) continue;
		NeuronID j = (*iter_post).i/patterns_every_post;
		if ( wrap_patterns ) 
			j = j % get_n_cols();
		if ( j < dst->get_size() && dst->localrank( j ) ) {
			for ( iter_pre = pattern1->begin() ; iter_pre != pattern1->end() ; ++iter_pre ) {
				if ( (*iter_pre).i%patterns_every_pre != 0 ) continue;
				NeuronID i = (*iter_pre).i/patterns_every_pre;
				if ( wrap_patterns ) 
					i = i % get_m_rows();
				if ( i < src->get_size() && w->exists( i, j ) ) { 
					if ( overwrite ) {
						set( i, j, (*iter_post).gamma*strength);
					} else {
						AurynWeight current_weight = get( i, j );
						set( i, j, current_weight + (*iter_post).gamma*strength);
					}
				}
			}
		}
	}
}

void SparseConnection::load_patterns( string filename, AurynWeight strength, bool overwrite, bool chainmode )
{
	load_patterns( filename, strength, 1000000, overwrite, chainmode );
}

void SparseConnection::load_patterns( string filename, AurynWeight strength, int n, bool overwrite, bool chainmode )
{

		ifstream fin (filename.c_str());
		if (!fin) {
			stringstream oss2;
			oss2 << "SparseConnection: There was a problem opening file " << filename << " for reading.";
			logger->msg(oss2.str(),WARNING);
			return;
		} else {
			stringstream oss;
			oss << "SparseConnection: ("<< get_name() <<"): Loading patterns from " << filename << " ...";
			logger->msg(oss.str(),NOTIFICATION);
		}

		unsigned int patcount = 0 ;

		NeuronID mindimension = min( get_m_rows()*patterns_every_pre, get_n_cols()*patterns_every_post );
		bool istoolarge = false;
		

		type_pattern pattern;
		vector<type_pattern> patterns;
		char buffer[256];
		string line;

		while(!fin.eof()) {
			line.clear();
			fin.getline (buffer,255);
			line = buffer;

			if(line[0] == '#') continue;
			if ( patcount >= n ) break;
			if (line == "") { 
				if ( pattern.size() > 0 ) {
					// put_pattern( &pattern, strength, overwrite );
					patterns.push_back(pattern);
					patcount++;
					pattern.clear();
				}
				continue;
			}

			pattern_member pm;
			stringstream iss (line);
			pm.gamma = 1 ; 
			iss >>  pm.i ;
			if ( !wrap_patterns && !istoolarge && pm.i > mindimension ) { 
				stringstream oss;
				oss << "SparseConnection: ("<< get_name() <<"): Some elements of pattern " << patcount << " are larger than the underlying NeuronGroups!";
				logger->msg(oss.str(),WARNING);
				istoolarge = true;
			}
			iss >>  pm.gamma ;
			if ( patterns_ignore_gamma ) 
				pm.gamma = 1;
			pattern.push_back(pm) ;
		}

		if ( chainmode ) {
			for ( int i = 0 ; i < patterns.size()-1 ; ++i ) {
				put_pattern( &(patterns[i]), &(patterns[i+1]), strength, overwrite );
			}
		} else {
			for ( int i = 0 ; i < patterns.size() ; ++i ) {
				put_pattern( &(patterns[i]), strength, overwrite );
			}
		}

		// put_pattern( &pattern, strength );

		fin.close();
		stringstream oss;
		oss << "SparseConnection: ("<< get_name() <<"): Added " << patcount << " patterns";
		logger->msg(oss.str(),NOTIFICATION);
}

vector<neuron_pair> SparseConnection::get_block(NeuronID lo_row, NeuronID hi_row,  NeuronID lo_col, NeuronID hi_col) 
{
	vector<neuron_pair> clist;
	for ( NeuronID i = 0 ; i < get_m_rows() ; ++i ) 
	{
		for ( NeuronID * j = w->get_row_begin(i) ; j != w->get_row_end(i) ; ++j )
		{
			if (i >= lo_row && i < hi_row && *j >= lo_col && *j < hi_col ) {
				neuron_pair a;
				a.i = i;
				a.j = *j;
				clist.push_back( a );
			}
		}
	}
	return clist;
}

vector<neuron_pair> SparseConnection::get_post_partners(NeuronID i) 
{
	vector<neuron_pair> clist;
	for ( NeuronID * j = w->get_row_begin(i) ; j != w->get_row_end(i) ; ++j )
	{
		neuron_pair a;
		a.i = i;
		a.j = *j;
		clist.push_back( a );
	}
	return clist;
}

vector<neuron_pair> SparseConnection::get_pre_partners(NeuronID j) 
{
	vector<neuron_pair> clist;
	for ( NeuronID i = 0 ; i < get_m_rows() ; ++i ) 
	{
		if ( get_ptr(i,j) != NULL ) {
			neuron_pair a;
			a.i = i;
			a.j = j;
			clist.push_back( a );
		}
	}
	return clist;
}


void SparseConnection::clip(AurynWeight lo, AurynWeight hi)
{
	for ( AurynWeight * ptr = w->get_data_begin() ; ptr != w->get_data_end()  ; ++ptr ) {
		if ( *ptr < lo )
			*ptr = lo;
		else
			if ( *ptr > hi )
				*ptr = hi;
	}
}
