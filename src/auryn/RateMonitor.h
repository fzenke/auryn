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

#ifndef RATEMONITOR_H_
#define RATEMONITOR_H_

#include <fstream>
#include <iomanip>

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Monitor.h"
#include "System.h"
#include "SpikingGroup.h"
#include "Trace.h"

namespace auryn {

/*! \brief Monitor class to record neural firing rates
 * 
 * Instances of this class record the firing rates of all neurons in the src SpikingGroup.
 * To estimate rates online it uses an exponential filter with a time constant that is
 * three times the sampling interval. To do so the class uses a synaptic trace. Each rank
 * only records spikes from local neurons.
 */

class RateMonitor : protected Monitor
{
private:
	/*! The sampling interval in units of AurynTime (auryn_timestep) */
	AurynTime ssize;

	/*! Filter time constant in seconds (by default 3x the sampling interval). */
	AurynDouble tau_filter;

	Trace * tr_post;

protected:
	/*! The source SpikingGroup */
	SpikingGroup * src;

	/*! Default init method */
	void init(SpikingGroup * source, string filename, AurynFloat samplinginterval);
	
public:

	/*! Default Constructor 
	 @param[source] The source spiking group.
	 @param[filename] The filename to write to (should be different for each rank.)
	 @param[samplinginterval] The sampling interval used for writing data to file.
	 which currently also determines the filter time constant. */
	RateMonitor(SpikingGroup * source, string filename, AurynFloat samplinginterval=1);

	/*! Default Destructor */
	virtual ~RateMonitor();

	/*! Implementation of necessary execute() function. */
	void execute();
};

}

#endif /*RATEMONITOR_H_*/
