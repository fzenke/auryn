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

#include "SparseConnection.h"

using namespace auryn;

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
		AurynDouble sparseness, 
		TransmitterType transmitter, 
		std::string name) 
	: Connection(source,destination,transmitter,name)
{
	init();
	std::stringstream oss;

	AurynLong anticipatedsize = (AurynLong) (estimate_required_nonzero_entires ( sparseness*src->get_pre_size()*dst->get_post_size() ) );
	oss << get_log_name() << "Assuming memory demand for pre #" << src->get_pre_size() << " and post #" << dst->get_post_size() 
													<< std::scientific << std::setprecision(4) << " ( total " << anticipatedsize << ")";
	auryn::logger->msg(oss.str(),VERBOSE);
	allocate(anticipatedsize);
	connect_random(weight,sparseness,skip_diagonal);
}


SparseConnection::SparseConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, 
		AurynDouble sparseness, 
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
		std::string name ) 
	: Connection(source,destination)
{
	set_transmitter(con->get_transmitter());
	AurynDouble sparseness = get_nonzero()/(con->get_m_rows()*con->get_n_cols());
	AurynDouble mean,std; con->stats(mean,std);
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
	auryn::sys->register_connection(this);
	has_been_allocated = false;
	if ( src == dst ) {
		skip_diagonal = true; 
		std::stringstream oss;
		oss << get_log_name() << "Detected recurrent connection. skip_diagonal was activated!";
		auryn::logger->msg(oss.str(),VERBOSE);
	}
	else skip_diagonal = false;

	if ( !has_been_seeded ) { // seed it only once 
		unsigned int rseed = sys->get_seed();
		seed(rseed);
	}

	set_min_weight(0.0);
	set_max_weight(std::numeric_limits<AurynWeight>::max()); // just make it large 

	patterns_ignore_gamma = false;
	wrap_patterns = false;

	patterns_every_pre = 1;
	patterns_every_post = 1;
}

void SparseConnection::seed(NeuronID randomseed) 
{
	std::stringstream oss;
	oss << get_log_name() << "Seeding with " << randomseed;
	auryn::logger->msg(oss.str(),VERBOSE);
	SparseConnection::sparse_connection_gen.seed(randomseed); 
	has_been_seeded = true;
}

AurynLong SparseConnection::estimate_required_nonzero_entires( AurynLong nonzero, double sigma )
{
	return std::min( (AurynLong)( nonzero + sigma*sqrt(1.0*nonzero) ), (AurynLong)(get_m_rows()*get_n_cols()) ) ;
}

void SparseConnection::free()
{
	delete w;
}


void SparseConnection::allocate(AurynLong bufsize)
{
	NeuronID m = get_m_rows();  
	NeuronID n = get_n_cols();

	std::stringstream oss;
	oss << get_log_name() << "Allocating sparse matrix (" << m << ", " << n << ") with space for "  << std::scientific << std::setprecision(4) << (double) bufsize <<  " nonzero elements ";

	auryn::logger->msg(oss.str(),VERBOSE);

	AurynLong maxsize = m*n;
	
	w = new ForwardMatrix ( m, n , std::min(maxsize,bufsize) );

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

void SparseConnection::set_max_weight(AurynWeight maximum_weight)
{
	wmax = maximum_weight;
}

void SparseConnection::random_data(AurynWeight mean, AurynWeight sigma) 
{
	random_data_normal(mean, sigma);
}

void SparseConnection::random_data_normal(AurynWeight mean, AurynWeight sigma) 
{
	std::stringstream oss;
	oss << get_log_name() << "randomizing non-zero connections (gaussian) with mean=" << mean << " sigma=" << sigma ;
	auryn::logger->msg(oss.str(),NOTIFICATION);

	boost::normal_distribution<> dist((double)mean, (double)sigma);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(SparseConnection::sparse_connection_gen, dist);
	AurynWeight rv;

	for ( AurynLong i = 0 ; i<w->get_nonzero() ; ++i ) {
		rv = die();
		if ( rv<get_min_weight() ) rv = get_min_weight();
		if ( rv>get_max_weight() ) rv = get_max_weight();
		w->set_data(i,rv);
	}
}

void SparseConnection::init_random_binary(AurynFloat prob, AurynWeight wlo, AurynWeight whi) 
{
	std::stringstream oss;
	oss << get_log_name() << "randomizing non-zero connections (gaussian) with binary weights between " 
		<< wlo << " and " << whi ;
	auryn::logger->msg(oss.str(),NOTIFICATION);

	boost::uniform_real<> dist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<> > die(SparseConnection::sparse_connection_gen, dist);
	AurynWeight rv;

	for ( AurynLong i = 0 ; i<w->get_nonzero() ; ++i ) {
		rv = die();
		if ( rv<prob ) {
			w->set_data(i,whi);
		} else {
			w->set_data(i,wlo);
		}
	}
}

void SparseConnection::sparse_set_data(AurynDouble sparseness, AurynWeight value) 
{
	std::stringstream oss;
	oss << get_log_name() << ": setting data sparsely with sparseness=" << sparseness << " value=" << value ;
	auryn::logger->msg(oss.str(),VERBOSE);

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
	std::stringstream oss;
	oss << get_log_name() << "Randomly scaling cols (gaussian) with mean=" << mean << " sigma=" << sigma ;
	auryn::logger->msg(oss.str(),VERBOSE);

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
	AurynWeight temp = std::max(weight,get_min_weight());
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
	AurynWeight temp = std::max(weight,get_min_weight());
	for ( NeuronID i = 0 ; i < get_m_rows() ; ++i ) 
	{
		for ( NeuronID * j = w->get_row_begin(i) ; j != w->get_row_end(i) ; ++j )
		{
			if ( i <=  *j )
			  w->get_data_begin()[j-w->get_row_begin(0)] = temp;
		}
	}
}


void SparseConnection::connect_block_random(AurynWeight weight, 
		AurynDouble sparseness,
		NeuronID lo_row, 
		NeuronID hi_row, 
		NeuronID lo_col, 
		NeuronID hi_col, 
		bool skip_diag )
{
	// do some sanity checks
	if ( sparseness <= 0.0 ) {
		auryn::logger->msg("Trying to set up a SparseConnection with sparseness smaller or equal to zero, which doesn't make sense",ERROR);
		throw AurynGenericException();
	}
	if ( sparseness > 1.0 ) {
		auryn::logger->msg("Sparseness larger than 1 not allowed. Setting to 1.",WARNING);
		sparseness = 1.0;
	}

	if ( weight < get_min_weight() ) {
		auryn::logger->msg("Weight smaller than minimal weight. Updating minimal weight and proceeding.",WARNING);
		set_min_weight(weight);
	}

	int r = 0; // these variables are used to speed up building the matrix if the destination is distributed
	int s = 1;

	r = sys->mpi_rank()-dst->get_locked_rank(); 
	s = dst->get_locked_range();

	// correction for "refractoriness"
	double lambda = 1.0/(1.0/sparseness-1.0);
	boost::exponential_distribution<> dist(lambda);
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(SparseConnection::sparse_connection_gen, dist);

	if (!has_been_allocated)
		throw AurynConnectionAllocationException();

	AurynLong idim = (hi_row-lo_row);
	AurynLong jdim = (hi_col-lo_col)/s;
	if ( (hi_col-lo_col)%s > r ) { // some ranks have one more "carry" neuron
		jdim += 1;
	}

	// we keep track of the real valued jump size ...
    AurynDouble jump = 0.0;
	// ... and the discretized position along the weight matrix
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
				std::stringstream oss;
				oss << get_log_name() 
					<<"Trying to add elements outside of matrix (i=" 
					<< i 
					<< "j="
					<< j 
					<< ", "
					<< count 
					<< "th element) ";
				auryn::logger->msg(oss.str(),ERROR);
				return;
			} 
			catch ( AurynMatrixPushBackException )
			{
				std::stringstream oss;
				oss << get_log_name() 
					<< "Failed pushing back element. Maybe due to out of order pushing? "
					<< " (" << i << "," << j << ") "
					<< " with count=" << count 
					<< " in connect_block_random ( fill_level= " << w->get_fill_level() << " )";
				auryn::logger->msg(oss.str(),ERROR);
				return;
			} 
			catch ( AurynMatrixBufferException )
			{
				std::stringstream oss;
				oss << get_log_name() 
					<<"Buffer full after pushing " 
					<< count 
					<< " elements."
					<< " There are pruned connections!";
				auryn::logger->msg(oss.str(),ERROR);
				return;
			} 
		}

		if ( sparseness < 1.0 ) { 
			// here we add our random exponential jump which was corrected
			// for the 1.0 "refractory period" to account for the fact that
			// we can have only one connection per "slot"
			jump += die()+1.0;

			// here we discretize this jump
			AurynLong discrete_jump = jump;

			// but keep the rest for the next jump to avoid discretization biases
			jump -= discrete_jump;

			// now we added discrete_jump to x and we are ready for the next connection
			x += discrete_jump;
		} else { // dense matrices
			x += 1 ;
		}
	}

	std::stringstream oss;
	oss << get_log_name() << "Finished connect_block_random ["
		<< lo_row << ", " << hi_row << ", " << lo_col << ", " << hi_col <<  "] " << " (stop count " 
		<< std::scientific << std::setprecision(4) << (double) stop 
		<< ") and successfully pushed " << (double) count <<  " entries. " 
    	<< "Resulting overall sparseness " << 1.*get_nonzero()/src->get_pre_size()/dst->get_post_size();
	auryn::logger->msg(oss.str(),VERBOSE);
}

void SparseConnection::connect_random(AurynWeight weight, AurynDouble sparseness, bool skip_diag)
{
	if ( dst->evolve_locally() ) { // if there are no local units there is no need for synapses
		std::stringstream oss;
		oss << get_log_name() <<"Randomfill with weight "<< weight <<  " and sparseness " << sparseness;
		auryn::logger->msg(oss.str(),VERBOSE);
		w->clear();
		connect_block_random(weight,sparseness,0,get_m_rows(),0,get_n_cols(),skip_diag);
	}
	finalize();
}

void SparseConnection::finalize()
{
	w->fill_zeros();
	if ( dst->evolve_locally() ) {
		std::stringstream oss;
		oss << get_log_name() << "Finalized with fill level " << w->get_fill_level();
		auryn::logger->msg(oss.str(),VERBOSE);
		if (w->get_fill_level()<WARN_FILL_LEVEL)
		{
			std::stringstream oss2;
			oss2 << get_log_name() <<"Wasteful fill level (" << w->get_fill_level() << ")! Make sure everything is in order!";
			auryn::logger->msg(oss2.str(),WARNING);
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
			std::stringstream oss;
			oss << "Sanity check: Neuron "
				<< i 
				<< " local (" 
				<< dst->global2rank(i) 
				<< ") has no inputs." ;
			auryn::logger->msg(oss.str(),WARNING);
		}
	}

	auryn::logger->parameter("sanity_check:total weight",total_weight);

	if ( unconnected_count ) { 
		std::stringstream oss;
		oss << "Sanity check failed ("
			<< get_name()
			<< "). Found " 
			<< unconnected_count
			<< " unconnected neurons.";
		auryn::logger->msg(oss.str(),WARNING);
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
			std::stringstream oss;
			oss << "Sanity check: Neuron "
			 << i 
			 << " local (" 
			 << src->global2rank(i) 
			 << ") has no outputs on this rank. This might be normal when run distributed." ;
			auryn::logger->msg(oss.str(),VERBOSE);
			if ( w->get_row_begin(i) != w->get_row_end(i) ) {
				auryn::logger->msg("wmat inconsistency",ERROR);
			}
		}
	}


	delete [] sum_rows;
}

void SparseConnection::stats(AurynDouble &mean, AurynDouble &std)
{
	stats(mean, std, 0);
}

void SparseConnection::stats(AurynDouble &mean, AurynDouble &std, NeuronID zid)
{
	double sum = 0; // needs double here -- machine precision really matters here
	double sum2 = 0;

	for ( AurynWeight * iter = w->get_data_begin(zid) ; iter != w->get_data_end(zid) ; ++iter ) {
		sum  += *iter;
		sum2 += (*iter * *iter);
	}

	NeuronID count = w->get_nonzero();

	if ( count <= 1 ) {
		mean = sum;
		std = 0;
		return;
	}

	mean = sum/count;
	std = sqrt((sum2-sum*sum/count)/(count-1));
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

void SparseConnection::set(std::vector<neuron_pair> element_list, AurynWeight value)
{
	for (std::vector<neuron_pair>::iterator iter = element_list.begin() ; iter != element_list.end() ; ++iter)
	{
		w->set((*iter).i, (*iter).j,value);
	}
}

void SparseConnection::set(NeuronID i, NeuronID j, AurynWeight value)
{
	value = std::max(value,get_min_weight());
	w->set(i,j,value);
}

bool SparseConnection::write_to_file(ForwardMatrix * m, std::string filename )
{
	if ( !dst->evolve_locally() ) return true;

	std::ofstream outfile;
	outfile.open(filename.c_str(),std::ios::out);
	if (!outfile) {
		std::stringstream oss;
	    oss << "Can't open output file " << filename;
		auryn::logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	outfile << "%%MatrixMarket matrix coordinate real general\n" 
		<< "% Auryn weight matrix. Has to be kept in row major order for load operation.\n" 
		<< "% Connection name: " << get_name() << "\n"
		<< "% Locked range: " << dst->get_locked_range() << "\n"
		<< "%\n"
		<< get_m_rows() << " " << get_n_cols() << " " << m->get_nonzero() << std::endl;

	AurynLong count = 0;
	for ( NeuronID i = 0 ; i < get_m_rows() ; ++i ) 
	{
		outfile << std::setprecision(7);
		for ( NeuronID * j = m->get_row_begin(i) ; j != m->get_row_end(i) ; ++j )
		{
			outfile << i+1 << " " << *j+1 << " " << std::scientific << m->get_data_begin()[j-m->get_row_begin(0)] << std::fixed << "\n";
			++count;
		}
	}

	if ( count != m->get_nonzero() ) {
		logger->msg("SparseConnection:: count inconsistency while writing MatrixMarket to file.", WARNING);
	}

	outfile.close();
	return true;
}

bool SparseConnection::write_to_file(std::string filename)
{
	return write_to_file(w,filename.c_str());
}

AurynLong SparseConnection::dryrun_from_file(std::string filename)
{
	if ( !dst->evolve_locally() ) return 0;

	char buffer[256];
	std::ifstream infile (filename.c_str());
	if (!infile) {
		std::stringstream oss;
		oss << get_log_name() << "Can't open input file " << filename;
		auryn::logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	NeuronID i,j;
	AurynLong k;
	unsigned int count = 0;
	unsigned int pushback_count = 0;
	float val;

	// read connection details
	infile.getline (buffer,256); count++;
	std::string header("%%MatrixMarket matrix coordinate real general");
	if (header.compare(buffer)!=0)
	{
		std::stringstream oss;
		oss << get_log_name() << "Input format not recognized.";
		auryn::logger->msg(oss.str(),ERROR);
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

bool SparseConnection::load_from_file(ForwardMatrix * m, std::string filename, AurynLong data_size )
{
	if ( !dst->evolve_locally() ) return true;

	char buffer[256];
	std::ifstream infile (filename.c_str());
	if (!infile) {
		std::stringstream oss;
		oss << get_log_name() << "Can't open input file " << filename;
		auryn::logger->msg(oss.str(),ERROR);
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
	std::string header("%%MatrixMarket matrix coordinate real general");
	if (header.compare(buffer)!=0)
	{
		std::stringstream oss;
		oss << get_log_name() << "Input format not recognized.";
		auryn::logger->msg(oss.str(),ERROR);
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
		std::stringstream oss;
		oss << get_log_name() << "Buffer too small ("
			<< m->get_datasize()
			<< " -> "
			<< k
			<< " elements). Reallocating.";
		auryn::logger->msg(oss.str() ,NOTIFICATION);
		m->resize_buffer_and_clear(k);
	}

	std::stringstream oss;
	oss << get_name() 
		<< ": Reading from file ("
		<< get_m_rows()<<"x"<<get_n_cols()
		<< " @ "<<1.*k/(src->get_size()*dst->get_rank_size())<<")";
	auryn::logger->msg(oss.str(),NOTIFICATION);


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
			std::stringstream oss;
			oss << get_log_name() << "Push back failed. Error in line=" << count << ", "
				<< " i=" << i 
				<< " j=" << j 
				<< " v=" << val << ". "
				<< " After pushing " << pushback_count << " elements. "
				<< " Bad row major order?";
			auryn::logger->msg(oss.str(),ERROR);
			throw AurynMMFileException();
			return false;
		} 
		catch ( AurynMatrixBufferException )
		{
			std::stringstream oss;
			oss << get_name() 
				<< ": Buffer full after pushing " 
				<< count << " elements."
				<< " There are pruned connections!";
			auryn::logger->msg(oss.str(),ERROR);
			return false;
		} 
		catch ( AurynMatrixDimensionalityException )
		{
			std::stringstream oss;
			oss << get_log_name() 
				<<"Trying to add elements outside of matrix (i=" 
				<< i 
				<< ", j="
				<< j 
				<< ", "
				<< count 
				<< "th element) ";
			auryn::logger->msg(oss.str(),ERROR);
			return false;
		} 
	}


	infile.close();

	if ( pushback_count != m->get_nonzero() ) { // this should never happen without an exception above, but better be save than sorry
		oss.str("");
		oss << get_log_name() 
			<< pushback_count 
			<< " elements pushed, but only "
			<< m->get_nonzero()
			<< " in matrix matrix.";
		auryn::logger->msg(oss.str(),ERROR);
	} else {
		oss.str("");
		oss << get_log_name() << "OK, " 
			<< pushback_count 
			<< " elements pushed.";
		auryn::logger->msg(oss.str(),VERBOSE);
	}

	m->fill_zeros();
	// finalize(); // commented this line out because it only acts on w

	return true;
}

bool SparseConnection::load_from_complete_file(std::string filename)
{
	AurynLong datasize = dryrun_from_file(filename);
	std::stringstream oss;
	oss << "Loading from complete wmat file "
		<< "(all ranks in the same file). Element count: "
		<< datasize 
		<< ".";
	auryn::logger->msg(oss.str(),NOTIFICATION);
	bool returnvalue = load_from_file(w,filename,datasize);
	finalize();
	return returnvalue;
}

bool SparseConnection::load_from_file(std::string filename)
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
	std::stringstream oss;
	oss << get_log_name() << "Putting assembly ( size " << pattern->size() << " )";
	auryn::logger->msg(oss.str(),VERBOSE);

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

void SparseConnection::load_patterns( std::string filename, AurynWeight strength, bool overwrite, bool chainmode )
{
	load_patterns( filename, strength, 1000000, overwrite, chainmode );
}

void SparseConnection::load_patterns( std::string filename, AurynWeight strength, int n, bool overwrite, bool chainmode )
{

		std::ifstream fin (filename.c_str());
		if (!fin) {
			std::stringstream oss2;
			oss2 << get_log_name() << "There was a problem opening file " << filename << " for reading.";
			auryn::logger->msg(oss2.str(),WARNING);
			return;
		} else {
			std::stringstream oss;
			oss << get_log_name() << "Loading patterns from " << filename << " ...";
			auryn::logger->msg(oss.str(),NOTIFICATION);
		}

		unsigned int patcount = 0 ;

		NeuronID mindimension = std::min( get_m_rows()*patterns_every_pre, get_n_cols()*patterns_every_post );
		bool istoolarge = false;
		

		type_pattern pattern;
		std::vector<type_pattern> patterns;
		char buffer[256];
		std::string line;

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
			std::stringstream iss (line);
			pm.gamma = 1 ; 
			iss >>  pm.i ;
			if ( !wrap_patterns && !istoolarge && pm.i > mindimension ) { 
				std::stringstream oss;
				oss << get_log_name() << "Some elements of pattern " << patcount << " are larger than the underlying NeuronGroups!";
				auryn::logger->msg(oss.str(),WARNING);
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
		std::stringstream oss;
		oss << get_log_name() << "Added " << patcount << " patterns";
		auryn::logger->msg(oss.str(),NOTIFICATION);
}

std::vector<neuron_pair> SparseConnection::get_block(NeuronID lo_row, NeuronID hi_row,  NeuronID lo_col, NeuronID hi_col) 
{
	std::vector<neuron_pair> clist;
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

std::vector<neuron_pair> SparseConnection::get_post_partners(NeuronID i) 
{
	std::vector<neuron_pair> clist;
	for ( NeuronID * j = w->get_row_begin(i) ; j != w->get_row_end(i) ; ++j )
	{
		neuron_pair a;
		a.i = i;
		a.j = *j;
		clist.push_back( a );
	}
	return clist;
}

std::vector<neuron_pair> SparseConnection::get_pre_partners(NeuronID j) 
{
	std::vector<neuron_pair> clist;
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
