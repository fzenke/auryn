/* 
* Copyright 2014 Friedemann Zenke
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
*/

#include "WeightMonitor.h"

WeightMonitor::WeightMonitor(SparseConnection * source, string filename, AurynDouble interval ) : Monitor(filename)
{
	init(source,0,0,filename,interval/dt);
	recordingmode = ELEMENTLIST;
	element_list = new vector<AurynWeight*>;
	elem_i = 0;
	elem_j = 0;
}

WeightMonitor::WeightMonitor(SparseConnection * source, ForwardMatrix * m, string filename, AurynDouble interval ) : Monitor(filename)
{
	init(source,0,0,filename,interval/dt);
	recordingmode = ELEMENTLIST;
	element_list = new vector<AurynWeight*>;
	elem_i = 0;
	elem_j = 0;
	set_mat(m);
}

WeightMonitor::WeightMonitor(SparseConnection * source, NeuronID i, NeuronID j, string filename, AurynDouble interval, RecordingMode mode ) : Monitor(filename)
{
	init(source,i,j,filename,interval/dt);
	recordingmode = mode;
	elem_i = i;
	elem_j = j;
	element_list = new vector<AurynWeight*>;

	switch (recordingmode) {
		case DATARANGE : 
			for (AurynLong i = elem_i ; i < elem_j ; ++i)
				add_to_list(mat->get_data_begin()+i) ;
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
	outfile << setiosflags(ios::fixed) << setprecision(6);

	stringstream oss;
	oss << "WeightMonitor:: "
		<< "Initialized. Writing to file "
		<< fname;
	logger->msg(oss.str(),DEBUG);
}

void WeightMonitor::add_to_list(AurynWeight * ptr)
{
	if ( recordingmode == ELEMENTLIST ) {
		if ( ptr != NULL )
			element_list->push_back( ptr );
	} else {
		stringstream oss;
		oss << "WeightMonitor:: "
			<< "Cannot add weight list. Not in ELEMENTLIST mode."
			<< endl;
		logger->msg(oss.str(),ERROR);
	}
}

void WeightMonitor::add_to_list(NeuronID i, NeuronID j)
{
	AurynWeight * ptr = mat->get_ptr(i,j);
	add_to_list(ptr);
}

void WeightMonitor::add_to_list( vector<neuron_pair>  vec, string label )
{
	if ( recordingmode == ELEMENTLIST ) {
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
			<< "Cannot add weight list. Not in ELEMENTLIST mode."
			<< endl;
		logger->msg(oss.str(),ERROR);
	}
}

void WeightMonitor::add_equally_spaced(NeuronID number)
{
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
		element_list->push_back( mat->get_data_begin()+a );
	outfile << "# Added data range " << i << "-" << j << "." << endl;
}

void WeightMonitor::load_pattern_connections( string filename , int maxcon, int maxpat, PatternMode patmod )
{
	if ( !src->get_destination()->evolve_locally() ) return;

	vector<type_pattern> patterns;

	ifstream fin (filename.c_str());
	if (!fin) {
		stringstream oss;
		oss << "WeightMonitor:: "
		<< "There was a problem opening file "
		<< filename
		<< " for reading."
		<< endl;
		logger->msg(oss.str(),ERROR);
		return;
	}

	char buffer[256];
	string line;

	type_pattern pattern;
	int total_pattern_size = 0;
	while( !fin.eof() && patterns.size() < maxpat ) {

		line.clear();
		fin.getline (buffer,255);
		line = buffer;

		if (line[0] == '#') continue;
		if (line == "") { 
			if ( total_pattern_size > 0 ) {
				stringstream oss;
				oss << "WeightMonitor:: Read pattern " 
					<< patterns.size() 
					<< " with pattern size "
					<< total_pattern_size
					<< " ( "
					<< pattern.size()
					<< " on rank )";
				logger->msg(oss.str(),DEBUG);

				patterns.push_back(pattern);
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

	for ( int i = 0 ; i < patterns.size() ; ++i ) {
		for ( int j = 0 ; j < patterns.size() ; ++j ) {
			if ( patmod==ASSEMBLIES_ONLY && i != j ) continue;
			vector<neuron_pair> list;
			for ( int k = 0 ; k < patterns[i].size() ; ++k ) {
				for ( int l = 0 ; l < patterns[j].size() ; ++l ) {
						neuron_pair p;
						p.i = patterns[i][k].i;
						p.j = patterns[j][l].i;
						AurynWeight * ptr = mat->get_ptr(p.i,p.j);
						if ( ptr != NULL ) // make sure we are counting connections that do exist
							list.push_back( p );
					if ( list.size() >= maxcon ) break;
				}
				if ( list.size() >= maxcon ) break;
			}

			stringstream oss;
			oss << "(connections " << i << " to " << j << ")";
			add_to_list(list,oss.str());
		}
	}


	stringstream oss;
	oss << "WeightMonitor:: Finished loading connections from " << patterns.size() << " patterns";
	logger->msg(oss.str(),NOTIFICATION);
}


void WeightMonitor::propagate()
{
	if ( src->get_destination()->evolve_locally() ) {
		if (sys->get_clock()%ssize==0) {
			outfile << fixed << dt*(sys->get_clock()) << scientific << " ";
			for (vector<AurynWeight*>::iterator iter = element_list->begin() ; iter != element_list->end() ; ++iter)
				outfile << *(*iter) << " ";
			outfile << "\n";
		}
	}
}

void WeightMonitor::set_mat(ForwardMatrix * m)
{
	mat = m;
}
