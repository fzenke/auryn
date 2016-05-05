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

#include "PatternStimulator.h"

using namespace auryn;

PatternStimulator::PatternStimulator(NeuronGroup * target, std::string filename, std::string patfile, AurynFloat scale, NeuronID maximum_patterns) : Monitor( "" )
{
	init(target,filename,scale,maximum_patterns);
	std::string s = patfile;
	if ( s.empty() ) {
		std::stringstream oss;
		auryn::logger->msg("No patfile given assuming whole population",NOTIFICATION);
		load_1_pattern();
	} else {
		load_patterns(patfile);
	}
}


PatternStimulator::~PatternStimulator()
{
	timeseriesfile.close();
	delete patterns;
	delete currents;
	delete newcurrents;
}

void PatternStimulator::init(NeuronGroup * target, std::string filename, AurynFloat scale, NeuronID maximum_patterns)
{
	auryn::sys->register_monitor(this);

	dst = target;
	set_scale(scale);
	maxpat = maximum_patterns;

	patterns = new std::vector<type_pattern>;

	mem = dst->mem;

	filetime = 0;
	timeseriesfile.open(filename.c_str(),std::ios::in);
	if (!timeseriesfile) {
	  std::stringstream oss;
	  oss << "Can't open time series file " << filename;
	  auryn::logger->msg(oss.str(),ERROR);
	  exit(1);
	}

}

void PatternStimulator::propagate()
{
	if ( dst->evolve_locally() ) {

		char buffer[256]; 
		std::string line;
		while( !timeseriesfile.eof() && filetime < auryn::sys->get_clock() ) {
			line.clear();
			timeseriesfile.getline (buffer,255); 
			// FIXME This buffer can quickly become to small of read lots of columns
			// Instead this section should be re-written to directly read one token/column 
			// at a time in the loop below. Just making the buffer size large should be avoided 
			// not to risk buffer overflow or unecessarily large memory consumption on cluster
			// machines with limited resources.
			line = buffer;
			if (line[0] == '#') continue;
			std::stringstream iss (line);
			double time;
			iss >> time;
			filetime = time/dt;

			for ( unsigned int column = 0 ; column < patterns->size() ; ++column ) {
				float cur;
				iss >> cur ;
				currents[column] = newcurrents[column];
				newcurrents[column] = cur;
			}
		}

		AurynState * cur_iter = currents;
		for ( std::vector<type_pattern>::const_iterator pattern = patterns->begin() ; 
				pattern != patterns->end() ; ++pattern ) { 
			for ( type_pattern::const_iterator piter = pattern->begin() ; piter != pattern->end() ; ++piter ) {
				mem->data[piter->i] += *cur_iter * piter->gamma * scl * dt;
			}
			++cur_iter;
		}
	}
}

void PatternStimulator::load_1_pattern( )
{
	type_pattern pattern;
	for ( unsigned int i = 0 ; i < dst->get_rank_size() ; ++i ) {
		pattern_member pm;
		pm.i = i;
		pm.gamma = 1 ; 
		pattern.push_back( pm ) ;
	}
	patterns->push_back(pattern);
	currents = new AurynState[1];
	newcurrents = new AurynState[1];
	currents[0] = 0;
	newcurrents[0] = 0;
}

void PatternStimulator::load_patterns( std::string filename )
{
	std::ifstream fin (filename.c_str());
	if (!fin) {
		std::stringstream oss;
		oss << "PatternStimulator:: "
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
				oss << "PatternStimulator:: Read pattern " 
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
		if ( dst->localrank( i ) ) {
			pattern_member pm;
			pm.gamma = 1 ; 
			iss >>  pm.gamma ;
			pm.i = dst->global2rank( i ) ;
			pattern.push_back( pm ) ;
		}
		total_pattern_size++;
	}

	fin.close();

	currents = new AurynState[patterns->size()];
	newcurrents = new AurynState[patterns->size()];
	for ( unsigned int i = 0 ; i < patterns->size() ; ++i ) {
		newcurrents[i] = 0;
		currents[i] = 0;
	}

	std::stringstream oss;
	oss << "PatternStimulator:: Finished loading " << patterns->size() << " patterns";
	auryn::logger->msg(oss.str(),NOTIFICATION);
}

void PatternStimulator::set_scale(AurynFloat scale) {
	scl = scale;
}

AurynFloat PatternStimulator::get_scale() {
	return scl;
}
