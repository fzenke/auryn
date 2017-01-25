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

#include "WeightMonitor.h"

using namespace auryn;

WeightMonitor::WeightMonitor(SparseConnection * source, std::string filename, AurynDouble interval ) : Monitor(filename)
{
	init(source,0,0,filename,interval/auryn_timestep);
}

WeightMonitor::WeightMonitor(SparseConnection * source, ForwardMatrix * m, std::string filename, AurynDouble interval ) : Monitor(filename)
{
	init(source,0,0,filename,interval/auryn_timestep);
	set_mat(m);
}

WeightMonitor::WeightMonitor(SparseConnection * source, NeuronID i, NeuronID j, std::string filename, AurynDouble interval, RecordingMode mode, StateID z ) : Monitor(filename)
{
	init(source,i,j,filename,interval/auryn_timestep);

	// overwrite the following default values set in init
	recordingmode = mode;
	elem_i = i;
	elem_j = j;

	switch (recordingmode) {
		case DATARANGE : 
			for (AurynLong c = elem_i ; c < elem_j ; ++c)
				add_to_list_by_data_index(c,z) ;
			break;
		case SINGLE :
			add_to_list(i,j,z);
			break;
		default:
		break;
	}
}

WeightMonitor::~WeightMonitor()
{
	delete element_list;

}

void WeightMonitor::init(SparseConnection * source, NeuronID i, NeuronID j, std::string filename, AurynTime stepsize)
{
	auryn::sys->register_device(this);
	src = source;
	set_mat(src->w);
	ssize = stepsize;
	if ( ssize < 1 ) ssize = 1;

	outfile << std::setiosflags(std::ios::fixed) << std::setprecision(6);

	std::stringstream oss;
	oss << "WeightMonitor:: "
		<< "Initialized. Writing to file "
		<< fname;
	auryn::logger->msg(oss.str(),VERBOSE);

	// default behavior
	recordingmode = ELEMENTLIST;
	element_list = new std::vector<AurynLong>;
	group_indices.push_back(0); // important for group mode
	elem_i = 0;
	elem_j = 0;
}

void WeightMonitor::add_to_list_by_data_index(AurynLong data_index, StateID z)
{
	element_list->push_back( data_index + z*mat->get_statesize() );
}

void WeightMonitor::add_to_list(AurynWeight * ptr)
{
	if ( ptr != NULL ) {
		element_list->push_back( mat->get_data_index(ptr) );
	}
}

void WeightMonitor::add_to_list(NeuronID i, NeuronID j, StateID z)
{
	if ( mat->exists(i, j, z) ) {
		add_to_list_by_data_index( mat->get_data_index(i, j), z );
	} else {
		std::stringstream oss;
		oss << "WeightMonitor:: Tried adding element " 
			<< i << ", " 
			<< j << " " 
			"z=" << z << " " 
			<< " but element does not exist";
		logger->msg(oss.str());
	}
}

void WeightMonitor::add_to_list( std::vector<neuron_pair>  vec, std::string label )
{
	if ( recordingmode == ELEMENTLIST || recordingmode == GROUPS ) {
		std::stringstream oss;
		oss << "WeightMonitor:: Adding " << vec.size() << " elements to index list " << label;
		auryn::logger->msg(oss.str(),VERBOSE);

		if ( label.empty() )
			outfile << "# Added list with " << vec.size() << " elements." << std::endl;
		else 
			outfile << "# Added list " << label << " with " << vec.size() << " elements." << std::endl;

		for (std::vector<neuron_pair>::iterator iter = vec.begin() ; iter != vec.end() ; ++iter)
		{
			add_to_list( (*iter).i, (*iter).j );
		}
	} else {
		std::stringstream oss;
		oss << "WeightMonitor:: "
			<< "Cannot add weight list. Not in ELEMENTLIST or GROUP mode."
			<< std::endl;
		auryn::logger->msg(oss.str(),ERROR);
	}
}

void WeightMonitor::add_equally_spaced(NeuronID number, NeuronID z)
{
	if ( z >= mat->get_num_z_values() ) {
		auryn::logger->msg("WeightMonitor:: z too large. Trying to monitor complex "
				"synaptic values which do not exist."
				,ERROR);
		return;
	}

	if ( number > src->get_nonzero() ) {
		auryn::logger->msg("WeightMonitor:: add_equally_spaced: "
        "Not enough elements in this Connection object",WARNING);
		number = src->get_nonzero();
	}

	for ( NeuronID i = 0 ; i < number ; ++i )
		add_to_list(mat->get_data_begin(z)+i*mat->get_nonzero()/number);

	std::stringstream oss;
	oss << "WeightMonitor:: "
		<< "Added "
		<< number 
		<< " equally spaced values.";
	auryn::logger->msg(oss.str(),VERBOSE);
}

void WeightMonitor::load_data_range( AurynLong i, AurynLong j )
{
	if ( !src->get_destination()->evolve_locally() ) return; 

	if ( j > mat->get_nonzero() )
		j = mat->get_nonzero();
	std::stringstream oss;
	oss << "WeightMonitor:: "
		<< "Adding data range i="
		<< i
		<< " j="
		<< j;
	auryn::logger->msg(oss.str(),VERBOSE);
	for ( NeuronID a = i ; a < j ; ++a )
		element_list->push_back( a );
	outfile << "# Added data range " << i << "-" << j << "." << std::endl;
}

std::vector<type_pattern> * WeightMonitor::load_patfile( std::string filename, unsigned int maxpat )
{

	std::vector<type_pattern> * patterns = new std::vector<type_pattern>;


	std::ifstream fin (filename.c_str());
	if (!fin) {
		std::stringstream oss;
		oss << "WeightMonitor:: "
		<< "There was a problem opening file "
		<< filename
		<< " for reading."
		<< std::endl;
		auryn::logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	char buffer[256];
	std::string line;


	type_pattern pattern;
	int total_pattern_size = 0;
	while( !fin.eof() && patterns->size() < maxpat ) {

		line.clear();
		fin.getline (buffer,255);
		line = buffer;

		if (line[0] == '#') continue;
		if (line == "") { 
			if ( total_pattern_size > 0 ) {
				std::stringstream oss;
				oss << "WeightMonitor:: Read pattern " 
					<< patterns->size() 
					<< " with pattern size "
					<< total_pattern_size
					<< " ( "
					<< pattern.size()
					<< " on rank )";
				auryn::logger->msg(oss.str(),VERBOSE);

				patterns->push_back(pattern);
				pattern.clear();
				total_pattern_size = 0;
			}
			continue;
		}

		std::stringstream iss (line);
		NeuronID i ;
		iss >> i ;
		pattern_member pm;
		pm.gamma = 1 ;
		iss >>  pm.gamma ;
		pm.i = i ;
		pattern.push_back( pm ) ;
		total_pattern_size++;
	}
	fin.close();



	return patterns;
}

void WeightMonitor::load_pattern_connections( std::string filename , unsigned int maxcon, unsigned int maxpat, PatternMode patmod )
{
	load_pattern_connections( filename, filename, maxcon, maxpat, patmod );
}


void WeightMonitor::load_pattern_connections( std::string filename_pre, std::string filename_post , unsigned int maxcon, unsigned int maxpat, PatternMode patmod )
{
	if ( !src->get_destination()->evolve_locally() ) return ;

	std::vector<type_pattern> * patterns_pre = load_patfile(filename_pre, maxpat);
	std::vector<type_pattern> * patterns_post = patterns_pre;

	if ( filename_pre.compare(filename_post) ) 
		patterns_pre = load_patfile(filename_post, maxpat);



	for ( unsigned int i = 0 ; i < patterns_pre->size() ; ++i ) {
		for ( unsigned int j = 0 ; j < patterns_post->size() ; ++j ) {
			if ( patmod==ASSEMBLIES_ONLY && i != j ) continue;
			std::vector<neuron_pair> list;
			for ( unsigned int k = 0 ; k < patterns_pre->at(i).size() ; ++k ) {
				for ( unsigned int l = 0 ; l < patterns_post->at(j).size() ; ++l ) {
						neuron_pair p;
						p.i = patterns_pre->at(i)[k].i;
						p.j = patterns_post->at(j)[l].i;
						AurynWeight * ptr =  mat->get_ptr(p.i,p.j);
						if ( ptr != NULL ) { // make sure we are counting connections that do exist
							list.push_back( p );
						}
					if ( list.size() >= maxcon ) break;
				}
				if ( list.size() >= maxcon ) break;
			}


			std::stringstream oss;
			oss << "(connections " << i << " to " << j << ")";
			add_to_list(list,oss.str());
			group_indices.push_back(element_list->size());
		}
	}



	std::stringstream oss;
	oss << "WeightMonitor:: Finished loading connections from n_pre=" 
		<< patterns_pre->size() 
		<< " and n_post="
		<< patterns_post->size() 
		<< " patterns";
	auryn::logger->msg(oss.str(),NOTIFICATION);


	if ( patterns_pre != patterns_post ) 
		delete patterns_post;
	delete patterns_pre;

}


void WeightMonitor::record_single_synapses()
{
	for (std::vector<AurynLong>::iterator iter = element_list->begin() ; iter != element_list->end() ; ++iter) {
		// the following is a workaround for the old WeightMonitor data_index ptr log to work with the new state vector 
		// based ComplexMatrix class
		// thus we need to translate the data index to a state variable z and the actual new data_index
		const AurynLong data_index = (*iter)%src->w->get_statesize();
		const StateID z = (*iter)/src->w->get_statesize();
		outfile << mat->get_data( data_index, z ) << " ";
	}
}

void WeightMonitor::record_synapse_groups()
{
	for ( unsigned int i = 1 ; i < group_indices.size() ; ++i ) {
		AurynDouble sum = 0;
		AurynDouble sum2 = 0;

		for ( unsigned int k = group_indices[i-1] ; k < group_indices[i] ; ++k ) {
			sum += mat->get_data( element_list->at(k) );
			sum2 += pow( mat->get_data( element_list->at(k) ), 2);
		}
		NeuronID n = group_indices[i]-group_indices[i-1];
		AurynDouble mean = sum/n;
		AurynDouble stdev = sqrt(sum2/n-mean*mean);
		outfile << mean << " " << stdev << "   ";
	}
}


void WeightMonitor::execute()
{
	if ( src->get_destination()->evolve_locally() ) {
		if (auryn::sys->get_clock()%ssize==0) {
			outfile << std::fixed << auryn_timestep*(auryn::sys->get_clock()) << std::scientific << " ";
			if ( recordingmode == GROUPS ) record_synapse_groups();
			else record_single_synapses();
			outfile << "\n";
		}
	}
}

void WeightMonitor::set_mat(ForwardMatrix * m)
{
	mat = m;
}
