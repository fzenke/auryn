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

#include "WeightMonitor.h"

WeightMonitor::WeightMonitor(SparseConnection * source, string filename, AurynDouble interval ) : Monitor(filename)
{
	init(source,0,0,filename,interval/dt);
}

WeightMonitor::WeightMonitor(SparseConnection * source, ForwardMatrix * m, string filename, AurynDouble interval ) : Monitor(filename)
{
	init(source,0,0,filename,interval/dt);
	set_mat(m);
}

WeightMonitor::WeightMonitor(SparseConnection * source, NeuronID i, NeuronID j, string filename, AurynDouble interval, RecordingMode mode ) : Monitor(filename)
{
	init(source,i,j,filename,interval/dt);

	// overwrite the following default values set in init
	recordingmode = mode;
	elem_i = i;
	elem_j = j;

	switch (recordingmode) {
		case DATARANGE : 
			for (AurynLong c = elem_i ; c < elem_j ; ++c)
				add_to_list(c) ;
			break;
		case SINGLE :
			add_to_list(i,j);
			break;
	}
}

WeightMonitor::~WeightMonitor()
{
	delete element_list;

}

void WeightMonitor::init(SparseConnection * source, NeuronID i, NeuronID j, string filename, AurynTime stepsize)
{
	sys->register_monitor(this);
	src = source;
	set_mat(src->w);
	ssize = stepsize;
	if ( ssize < 1 ) ssize = 1;

	outfile << setiosflags(ios::fixed) << setprecision(6);

	stringstream oss;
	oss << "WeightMonitor:: "
		<< "Initialized. Writing to file "
		<< fname;
	logger->msg(oss.str(),DEBUG);

	// default behavior
	recordingmode = ELEMENTLIST;
	element_list = new vector<AurynLong>;
	group_indices.push_back(0); // important for group mode
	elem_i = 0;
	elem_j = 0;
}

void WeightMonitor::add_to_list(AurynLong data_index)
{
	element_list->push_back( data_index );
}

void WeightMonitor::add_to_list(AurynWeight * ptr)
{
	if ( ptr != NULL ) {
		element_list->push_back( mat->get_data_index(ptr) );
	}
}

void WeightMonitor::add_to_list(NeuronID i, NeuronID j)
{
	add_to_list( mat->get_data_index(i,j) );
}

void WeightMonitor::add_to_list( vector<neuron_pair>  vec, string label )
{
	if ( recordingmode == ELEMENTLIST || recordingmode == GROUPS ) {
		stringstream oss;
		oss << "WeightMonitor:: Adding " << vec.size() << " elements to index list " << label;
		logger->msg(oss.str(),DEBUG);

		if ( label.empty() )
			outfile << "# Added list with " << vec.size() << " elements." << endl;
		else 
			outfile << "# Added list " << label << " with " << vec.size() << " elements." << endl;

		for (vector<neuron_pair>::iterator iter = vec.begin() ; iter != vec.end() ; ++iter)
		{
			add_to_list( (*iter).i, (*iter).j );
		}
	} else {
		stringstream oss;
		oss << "WeightMonitor:: "
			<< "Cannot add weight list. Not in ELEMENTLIST or GROUP mode."
			<< endl;
		logger->msg(oss.str(),ERROR);
	}
}

void WeightMonitor::add_equally_spaced(NeuronID number)
{
	if ( number > src->get_nonzero() ) {
		logger->msg("WeightMonitor:: add_equally_spaced: \
				Not enough elements in this Connection object",WARNING);
		number = src->get_nonzero();
	}

	for ( NeuronID i = 0 ; i < number ; ++i )
		add_to_list(mat->get_data_begin()+i*mat->get_nonzero()/number);

	stringstream oss;
	oss << "WeightMonitor:: "
		<< "Adding "
		<< number 
		<< " equally spaced values.";
	logger->msg(oss.str(),DEBUG);
}

void WeightMonitor::load_data_range( NeuronID i, NeuronID j )
{
	if ( !src->get_destination()->evolve_locally() ) return; 

	if ( j > mat->get_nonzero() )
		j = mat->get_nonzero();
	stringstream oss;
	oss << "WeightMonitor:: "
		<< "Adding data range i="
		<< i
		<< " j="
		<< j;
	logger->msg(oss.str(),DEBUG);
	for ( NeuronID a = i ; a < j ; ++a )
		element_list->push_back( a );
	outfile << "# Added data range " << i << "-" << j << "." << endl;
}

vector<type_pattern> * WeightMonitor::load_patfile( string filename, int maxpat )
{

	vector<type_pattern> * patterns = new vector<type_pattern>;


	ifstream fin (filename.c_str());
	if (!fin) {
		stringstream oss;
		oss << "WeightMonitor:: "
		<< "There was a problem opening file "
		<< filename
		<< " for reading."
		<< endl;
		logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	char buffer[256];
	string line;


	type_pattern pattern;
	int total_pattern_size = 0;
	while( !fin.eof() && patterns->size() < maxpat ) {

		line.clear();
		fin.getline (buffer,255);
		line = buffer;

		if (line[0] == '#') continue;
		if (line == "") { 
			if ( total_pattern_size > 0 ) {
				stringstream oss;
				oss << "WeightMonitor:: Read pattern " 
					<< patterns->size() 
					<< " with pattern size "
					<< total_pattern_size
					<< " ( "
					<< pattern.size()
					<< " on rank )";
				logger->msg(oss.str(),DEBUG);

				patterns->push_back(pattern);
				pattern.clear();
				total_pattern_size = 0;
			}
			continue;
		}

		stringstream iss (line);
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

void WeightMonitor::load_pattern_connections( string filename , int maxcon, int maxpat, PatternMode patmod )
{
	load_pattern_connections( filename, filename, maxcon, maxpat, patmod );
}


void WeightMonitor::load_pattern_connections( string filename_pre, string filename_post , int maxcon, int maxpat, PatternMode patmod )
{
	if ( !src->get_destination()->evolve_locally() ) return ;

	vector<type_pattern> * patterns_pre = load_patfile(filename_pre, maxpat);
	vector<type_pattern> * patterns_post = patterns_pre;

	if ( filename_pre.compare(filename_post) ) 
		patterns_pre = load_patfile(filename_post, maxpat);



	for ( int i = 0 ; i < patterns_pre->size() ; ++i ) {
		for ( int j = 0 ; j < patterns_post->size() ; ++j ) {
			if ( patmod==ASSEMBLIES_ONLY && i != j ) continue;
			vector<neuron_pair> list;
			for ( int k = 0 ; k < patterns_pre->at(i).size() ; ++k ) {
				for ( int l = 0 ; l < patterns_post->at(j).size() ; ++l ) {
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


			stringstream oss;
			oss << "(connections " << i << " to " << j << ")";
			add_to_list(list,oss.str());
			group_indices.push_back(element_list->size());
		}
	}



	stringstream oss;
	oss << "WeightMonitor:: Finished loading connections from n_pre=" 
		<< patterns_pre->size() 
		<< " and n_post="
		<< patterns_post->size() 
		<< " patterns";
	logger->msg(oss.str(),NOTIFICATION);


	if ( patterns_pre != patterns_post ) 
		delete patterns_post;
	delete patterns_pre;

}


void WeightMonitor::record_single_synapses()
{
	for (vector<AurynLong>::iterator iter = element_list->begin() ; iter != element_list->end() ; ++iter)
		outfile << mat->get_data( (*iter) ) << " ";
}

void WeightMonitor::record_synapse_groups()
{
	for ( int i = 1 ; i < group_indices.size() ; ++i ) {
		AurynDouble sum = 0;
		AurynDouble sum2 = 0;

		for ( int k = group_indices[i-1] ; k < group_indices[i] ; ++k ) {
			sum += mat->get_data( element_list->at(k) );
			sum2 += pow( mat->get_data( element_list->at(k) ), 2);
		}
		NeuronID n = group_indices[i]-group_indices[i-1];
		AurynDouble mean = sum/n;
		AurynDouble stdev = sqrt(sum2/n-mean*mean);
		outfile << mean << " " << stdev << "   ";
	}
}


void WeightMonitor::propagate()
{
	if ( src->get_destination()->evolve_locally() ) {
		if (sys->get_clock()%ssize==0) {
			outfile << fixed << dt*(sys->get_clock()) << scientific << " ";
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
