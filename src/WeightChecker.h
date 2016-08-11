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

#ifndef WEIGHTCHECKER_H_
#define WEIGHTCHECKER_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "Checker.h"
#include "Connection.h"

namespace auryn {

/*! \brief A Checker class that tracks the meain weight of a Connection 
 * and breaks a run if it goes out of bound.
 *
 */

class WeightChecker : public Checker
{
private:
	AurynWeight wmin;
	AurynWeight wmax;
	AurynTime timestep_;
	Connection * source_;
	void init(Connection * source, AurynFloat min, AurynFloat max, AurynFloat timestep);
	
public:
	/*! \brief The default constructor.
	 *
	 * @param source the source group to monitor.
	 * @param max the maximum mean weight above which the Checker signals a break of the simulation.
	 */
	WeightChecker(Connection * source, AurynFloat max);
	/*! \brief A more elaborate constructor specifying also a minimum weight to guard 
	 * against silent networks.
	 *
	 * @param source the source group to monitor.
	 * @param min the minimum mean weight below which the Checker signals a break of the simulation.
	 * @param max the maximum mean weight above which the Checker signals a break of the simulation.
	 * @param timestep Check weights every timestep seconds (default = 10s because weight checking is expensive).
	 */
	WeightChecker(Connection * source, AurynFloat min, AurynFloat max, AurynFloat timestep=10.0);
	virtual ~WeightChecker();
	/*! The propagate function required for internal use. */
	virtual bool propagate();
};

}

#endif /*WEIGHTCHECKER_H_*/
