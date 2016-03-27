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

#include "PatternMonitor.h"

using namespace auryn;

PatternMonitor::PatternMonitor(SpikingGroup * source, std::string filename, std::string patfile, NeuronID maximum_patterns, AurynFloat binsize) : Monitor(filename)
{
	init(source,filename,maximum_patterns,binsize);
	load_patterns(patfile);
}

PatternMonitor::PatternMonitor(SpikingGroup * source, std::string filename, StimulusGroup * stimgroup, NeuronID maximum_patterns, AurynFloat binsize) : Monitor(filename)
{
	init(source,filename,maximum_patterns,binsize);
	delete patterns;
	linked_to_stimgroup = true;
	patterns = stimgroup->get_patterns();

	std::stringstream oss;
	oss << "PatternMonitor:: Linked to StimulusGroup";
	auryn::logger->msg(oss.str(),NOTIFICATION);
}

PatternMonitor::~PatternMonitor()
{
	delete [] counter;
	if ( !linked_to_stimgroup )
		delete patterns;
}

void PatternMonitor::init(SpikingGroup * source, std::string filename, NeuronID maximum_patterns, AurynFloat binsize)
{
	auryn::sys->register_monitor(this);

	linked_to_stimgroup = false;

	src = source;
	bsize = binsize;
	ssize = bsize/dt;
	if ( ssize < 1 ) ssize = 1;

	maxpat = maximum_patterns;

	counter = new  NeuronID [src->get_rank_size()];
	patterns = new std::vector<type_pattern>;

	for ( int i = 0 ; i < src->get_rank_size() ; ++i )
		counter[i] = 0;

	std::stringstream oss;
	oss << "PatternMonitor:: Setting binsize " << bsize << "s";
	auryn::logger->msg(oss.str(),NOTIFICATION);

	outfile << std::setiosflags(std::ios::fixed) << std::setprecision(6);
}

void PatternMonitor::propagate()
{
	if ( src->evolve_locally() ) {
		for ( SpikeContainer::const_iterator iter = src->get_spikes_immediate()->begin() ;
				iter != src->get_spikes_immediate()->end() ; ++iter ) {
			counter[src->global2rank(*iter)] += 1; // might be more memory efficent to count here in rankIDs (local)
		}

		if (auryn::sys->get_clock()%ssize==0) {
			outfile << dt*(auryn::sys->get_clock());
			for ( std::vector<type_pattern>::iterator pattern = patterns->begin() ; 
					pattern != patterns->end() ; ++pattern ) { 
				double sum = 0.;
				NeuronID act = 0;
				for ( type_pattern::iterator piter = pattern->begin() ; piter != pattern->end() ; ++piter ) {
					if ( counter[piter->i] > 0 ) {
						sum += counter[piter->i];
						act++;
					}
				}
				double mean = sum/pattern->size();
				outfile << " " << mean/bsize << " " << 1.0*act/pattern->size();
			}

			NeuronID act = 0;
			for ( int i = 0 ; i < src->get_rank_size() ; ++i )
				if (counter[i]) act++;

			outfile << " " << 1.0*act/src->get_rank_size() << "\n"; // total activation

			// reset counter
			for ( int i = 0 ; i < src->get_rank_size() ; ++i )
				counter[i] = 0;
		}
	}
}

void PatternMonitor::load_patterns( std::string filename )
{

		// if ( not src->evolve_locally() ) return ;

	std::ifstream fin (filename.c_str());
	if (!fin) {
		std::stringstream oss;
		oss << "PatternMonitor:: "
		<< "There was a problem opening file "
		<< filename
		<< " for reading."
		<< std::endl;
		auryn::logger->msg(oss.str(),ERROR);
		return;
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
				oss << "PatternMonitor:: Read pattern " 
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
		if ( src->localrank( i ) ) {
			pattern_member pm;
			pm.gamma = 1 ; 
			iss >>  pm.gamma ;
			pm.i = src->global2rank( i ) ;
			pattern.push_back( pm ) ;
		}
		total_pattern_size++;
	}

	fin.close();

	std::stringstream oss;
	oss << "PatternMonitor:: Finished loading " << patterns->size() << " patterns";
	auryn::logger->msg(oss.str(),NOTIFICATION);
}

void PatternMonitor::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	for ( NeuronID i = 0 ; i < src->get_rank_size() ; ++i )
		ar & counter[i] ;
}

void PatternMonitor::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	for ( NeuronID i = 0 ; i < src->get_rank_size() ; ++i )
		ar & counter[i] ;
}
