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

#ifndef WEIGHTPATTERNMONITOR_H_
#define WEIGHTPATTERNMONITOR_H_

#include "auryn_definitions.h"
#include "Monitor.h"
#include "System.h"
#include "Connection.h"
#include <fstream>
#include <iomanip>

using namespace std;

/*! \brief Records mean weights from a connection specified by one or two
 *  pattern files. Can be used to easily monitor the mean synaptic weight
 *  in assemblies or feed-forward connections of populations of neurons.
 */
class WeightPatternMonitor : protected Monitor
{
protected:
	Connection * src;
	AurynTime ssize;
	NeuronID data_size_limit;

	vector<type_pattern> pre_patterns;
	vector<type_pattern> post_patterns;


	void init(Connection * source, string filename, AurynTime stepsize);
	AurynWeight compute_pattern_mean(const NeuronID i, const NeuronID j);

	/*! Mother function for loading patterns */
	void load_patterns(string filename, vector<type_pattern> & patterns );
	
public:
	WeightPatternMonitor(Connection * source, string filename, AurynDouble binsize=10.0);

	/*! Loads pre patterns for asymmetric monitoring. Each pre pattern needs to be matched
	 * by a corresponding post pattern otherwise there will be a crash. */
	void load_pre_patterns(string filename );
	/*! Loads post patterns for asymmetric monitoring. Each post pattern needs to be matched
	 * by a corresponding pre pattern otherwise there will be a crash. */
	void load_post_patterns(string filename );
	/*! Loads patterns for symmetric assembly monitoring */
	void load_patterns(string filename );

	virtual ~WeightPatternMonitor();
	void propagate();
};

#endif /*WEIGHTPATTERNMONITOR_H_*/
