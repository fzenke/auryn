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

#ifndef PATTERNSTIMULATOR_H_
#define PATTERNSTIMULATOR_H_

#include "auryn_definitions.h"
#include "System.h"
#include "Monitor.h"
#include "NeuronGroup.h"
#include <fstream>
#include <iomanip>

namespace auryn {

/*! \brief Stimulator class to inject timeseries of currents to patterns (subpopulations) of neurons 
 * 
 * Instances of this class inject currents that vary over time to subpopulations of the NeuronGroup assigned.
 */

class PatternStimulator : protected Monitor
{
private:
	/*! Maximum number of patterns to inject to */
	NeuronID maxpat;
	/*! Vector storing all the patterns */
	std::vector<type_pattern> * patterns;
	/*! Vector storing all the current values */
	AurynState * currents;
	/*! Vector storing all new current values */
	AurynState * newcurrents;
	/*! Input filestream for time series data */
	std::ifstream timeseriesfile;
	/*! Target membrane */
	auryn_vector_float * mem;

	/*! Scale stimulus size */
	AurynFloat scl;

	/*! Stores time in the file */
	AurynTime filetime;


protected:
	/*! The target NeuronGroup */
	NeuronGroup * dst;
	/*! Default init method */
	void init(NeuronGroup * target, string filename, AurynFloat scale, NeuronID maximum_patterns);
	
public:
	/*! Default Constructor 
	 @param[target] The target spiking group.
	 @param[filename] The filename to read the timeseries from 
	 @param[patfile] Filename from where to load the patterns */
	PatternStimulator(NeuronGroup * target, 
			string filename, 
			string patfile="", 
			AurynFloat scale=1,
			NeuronID maximum_patterns=10);
	/*! Default Destructor */
	virtual ~PatternStimulator();
	void set_scale(AurynFloat scale);
	AurynFloat get_scale();
	/*! Implementation of necessary propagate() function. */
	void propagate();
	/*! Load patterns from file */
	void load_patterns( string filename );

	/*! Loads the 1111 pattern */
	void load_1_pattern(  );
};

}

#endif /*PATTERNSTIMULATOR_H_*/
