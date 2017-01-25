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

#ifndef WEIGHTMATRIXMONITOR_H_
#define WEIGHTMATRIXMONITOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Monitor.h"
#include "System.h"
#include "Connection.h"
#include <fstream>
#include <iomanip>

namespace auryn {

/*! \brief Saves the weight matrix of a given connection in regular time intervals.
 */
class WeightMatrixMonitor : protected Monitor
{
protected:
	Connection * src;
	AurynTime ssize;
	AurynInt filecount;
	void init(Connection * source, AurynFloat stepsize);
	
public:
	WeightMatrixMonitor(Connection * source, string filename, AurynFloat stepsize=600.);
	virtual ~WeightMatrixMonitor();
	void execute();
};

}

#endif /*WEIGHTMATRIXMONITOR_H_*/
