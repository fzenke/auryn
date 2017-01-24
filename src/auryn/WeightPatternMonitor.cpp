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

#include "WeightPatternMonitor.h"

using namespace auryn;

WeightPatternMonitor::WeightPatternMonitor(Connection * source, std::string filename, AurynDouble binsize) : Monitor(filename)
{
	init(source,filename,binsize/auryn_timestep);
}

WeightPatternMonitor::~WeightPatternMonitor()
{
}

void WeightPatternMonitor::init(Connection * source, std::string filename,AurynTime stepsize)
{
	if ( !source->get_destination()->evolve_locally() ) return;

	auryn::sys->register_device(this);

	src = source;
	ssize = stepsize;
	if ( ssize < 1 ) ssize = 1;

	outfile << std::setiosflags(std::ios::fixed) << std::setprecision(6);

	max_patterns = 5;
}


AurynWeight WeightPatternMonitor::compute_pattern_mean(const NeuronID i, const NeuronID j)
{
	AurynWeight sum = 0;
	// AurynFloat sum2 = 0;
	NeuronID n = 0;
	for ( NeuronID k = 0 ; k < pre_patterns[i].size() ; ++k ) {
		for ( NeuronID l = 0 ; l < post_patterns[j].size() ; ++l ) {
			pattern_member p = pre_patterns[i][k];
			pattern_member q = post_patterns[j][l];
			AurynWeight * val = src->get_ptr(p.i,q.i);
			if ( val ) {
				sum += *val;
				// sum2 += *val* *val;
				n++;
			}
		}
	}
	AurynFloat mean = sum/n;
	return mean;
}

void WeightPatternMonitor::execute()
{
	if (auryn::sys->get_clock()%ssize==0) {
		outfile << std::fixed << (auryn::sys->get_time()) << " ";

		int p = std::min(std::min(pre_patterns.size(),post_patterns.size()),max_patterns);

		for ( int i = 0 ; i < p ; ++i ) {
			for ( int j = 0 ; j < p ; ++j ) {
				outfile << std::scientific 
					<< compute_pattern_mean(i,j) 
					<< " ";
			}
		}
		outfile << std::endl;
	}
}


void WeightPatternMonitor::load_patterns( std::string filename, std::vector<type_pattern> & patterns )
{
	std::ifstream fin (filename.c_str());
	if (!fin) {
		std::stringstream oss;
		oss << "WeightPatternMonitor:: "
		<< "There was a problem opening file "
		<< filename
		<< " for reading."
		<< std::endl;
		auryn::logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	char buffer[256];
	std::string line;

	patterns.clear();

	type_pattern pattern;
	int total_pattern_size = 0;
	while(!fin.eof()) {

		line.clear();
		fin.getline (buffer,255);
		line = buffer;

		if (line[0] == '#') continue;
		if (line == "") { 
			if ( total_pattern_size > 0 ) {
				std::stringstream oss;
				oss << "WeightPatternMonitor:: Read pattern " 
					<< patterns.size() 
					<< " with pattern size "
					<< total_pattern_size
					<< " ( "
					<< pattern.size()
					<< " on rank )";
				auryn::logger->msg(oss.str(),VERBOSE);

				patterns.push_back(pattern);
				pattern.clear();
				total_pattern_size = 0;
			}
			continue;
		}

		std::stringstream iss (line);
		NeuronID i ;
		iss >> i ;
		pattern_member pm;
		pm.gamma = 1.0 ;
		iss >>  pm.gamma ;
		pm.i = i ;
		pattern.push_back( pm ) ;
		total_pattern_size++;
	}

	fin.close();

	std::stringstream oss;
	oss << "WeightPatternMonitor:: Finished loading " << patterns.size() << " patterns";
	auryn::logger->msg(oss.str(),NOTIFICATION);
}

void WeightPatternMonitor::load_pre_patterns( std::string filename ) 
{
	load_patterns(filename,pre_patterns);
}

void WeightPatternMonitor::load_post_patterns( std::string filename ) 
{
	load_patterns(filename,post_patterns);
}

void WeightPatternMonitor::load_patterns( std::string filename ) 
{
	load_pre_patterns(filename);
	load_post_patterns(filename);
}
