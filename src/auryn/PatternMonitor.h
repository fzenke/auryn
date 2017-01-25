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

#ifndef PATTERNMONITOR_H_
#define PATTERNMONITOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Monitor.h"
#include "System.h"
#include "SpikingGroup.h"
#include "StimulusGroup.h"
#include <fstream>
#include <iomanip>

namespace auryn {

/*! \brief Monitor class to record population firing rates
 * 
 * Instances of this class record the population firing rate of the src SpikingGroup assigned.
 * Binning is done discretely in bins of size bsize that is directly transformed in discrete 
 * AurynTime steps. The default 
 */

class PatternMonitor : protected Monitor
{
private:
	/*! Array used to count the spike events of the src SpikingGroup */
	NeuronID * counter;
	/*! Stepsize = binsize in units of AurynTime (auryn_timestep) */
	AurynTime ssize;
	/*! Binsize used in seconds */
	AurynDouble bsize;
	/*! Maximum number of patterns to record from */
	NeuronID maxpat;
	/*! Vector storing all the patterns */
	std::vector<type_pattern> * patterns;

	bool linked_to_stimgroup;

protected:
	/*! The source SpikingGroup */
	SpikingGroup * src;
	/*! Default init method */
	void init(SpikingGroup * source, string filename, NeuronID maximum_patterns, AurynFloat binsize);

	virtual void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version );
	virtual void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version );
	
public:
	/*! Default Constructor 
	 @param[source] The source spiking group.
	 @param[filename] The filename to write to (should be different for each rank.)
	 @param[patfile] Filename from where to load the patterns
	 @param[binsize] The binsize used for counting in seconds.*/
	PatternMonitor(SpikingGroup * source, string filename, 
			string patfile, 
			NeuronID maximum_patterns=10, 
			AurynFloat binsize=1e-1);
	/*! Constructor that loads patterns from a StimulusGroup
	 @param[source] The source spiking group.
	 @param[filename] The filename to write to (should be different for each rank.)
	 @param[stimgroup] Stimulus group to load patterns from
	 @param[binsize] The binsize used for counting in seconds.*/
	PatternMonitor(SpikingGroup * source, string filename, 
			StimulusGroup * stimgroup,
			NeuronID maximum_patterns=10, 
			AurynFloat binsize=1e-1);
	/*! Default Destructor */
	virtual ~PatternMonitor();
	/*! Implementation of necessary execute() function. */
	void execute();
	/*! Load patterns from file */
	void load_patterns( string filename );
};

}

#endif /*PATTERNMONITOR_H_*/
