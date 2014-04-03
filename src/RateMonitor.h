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

#ifndef RATEMONITOR_H_
#define RATEMONITOR_H_

#include "auryn_definitions.h"
#include "Monitor.h"
#include "System.h"
#include "SpikingGroup.h"
#include <fstream>
#include <iomanip>

using namespace std;

/*! \brief Monitor class to record neural firing rates
 * 
 * Instances of this class record the population firing rate of the src SpikingGroup assigned.
 * Binning is done discretely in bins of size bsize that is directly transformed in discrete 
 * AurynTime steps. The default 
 */

class RateMonitor : protected Monitor
{
private:
	/*! The sampling interval in units of AurynTime (dt) */
	AurynTime ssize;

	/*! Filter time constant in seconds (by default 3x the sampling interval). */
	AurynDouble tau_filter;

protected:
	/*! The source SpikingGroup */
	SpikingGroup * src;

	/*! Trace varible used to count the spike events of the src SpikingGroup */
	DEFAULT_TRACE_MODEL * tr_post;

	/*! Default init method */
	void init(SpikingGroup * source, string filename, AurynFloat samplinginterval);
	
public:

	/*! Default Constructor 
	 @param[source] The source spiking group.
	 @param[filename] The filename to write to (should be different for each rank.)
	 @param[samplinginterval] The sampling interval used for writing data to file.*/
	RateMonitor(SpikingGroup * source, string filename, AurynFloat samplinginterval=1);

	/*! Default Destructor */
	virtual ~RateMonitor();

	/*! Implementation of necessary propagate() function. */
	void propagate();
};

#endif /*RATEMONITOR_H_*/
