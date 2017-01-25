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

#ifndef REALTIMEMONITOR_H_
#define REALTIMEMONITOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Monitor.h"
#include "System.h"
#include "SpikingGroup.h"
#include <fstream>
#include <iomanip>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace auryn {

/*! \brief Monitor class to record the system time in every timestep
 * 
 * The RealTimeMonitor records every timestep RealTime (boost us clock) vs AurynTime
 */

class RealTimeMonitor : protected Monitor
{
private:
	/*! Start time */
	AurynTime t_start;

	/*! Stop time */
	AurynTime t_stop;

	/*! Time offset to keep filesize smaller */
	boost::posix_time::ptime ptime_offset ;

protected:
	
public:
	/*! Default Constructor 
	 @param[filename] The filename to write to (should be different for each rank.)
	 @param[start] Start time.)
	 @param[stop] Stop time.*/
	RealTimeMonitor(std::string filename, AurynDouble start = 1e-3, AurynDouble stop = 100);

	/*! Default Destructor */
	virtual ~RealTimeMonitor();

	/*! Implementation of necessary execute() function. */
	void execute();
};

}

#endif /*REALTIMEMONITOR_H_*/
