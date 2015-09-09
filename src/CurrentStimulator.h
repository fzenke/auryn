/* 
* Copyright 2014-2015 Friedemann Zenke
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

#ifndef CURRENTSTIMULATOR_H_
#define CURRENTSTIMULATOR_H_

#include "auryn_definitions.h"
#include "System.h"
#include "Logger.h"
#include "Monitor.h"
#include "NeuronGroup.h"


using namespace std;

/*! \brief Stimulator class to inject timeseries of currents to patterns (subpopulations) of neurons 
 * 
 * Instances of this class inject currents that vary over time to subpopulations of the NeuronGroup assigned.
 */

class CurrentStimulator : protected Monitor
{
private:

	/*! Vector storing all the current values */
	auryn_vector_float * currents;

	/*! Target membrane */
	auryn_vector_float * target_vector;

	/*! Default init method */
	void init(NeuronGroup * target, AurynFloat initial_current );

	void free();

	/*! Returns the lambda parameter of the pmf for Current. */
	AurynFloat get_lambda();

protected:

	/*! The target NeuronGroup */
	NeuronGroup * dst;

	
public:
	/*! Default Constructor 
	 * @param[initial_current] Initializes all currents with this value
	 */
	CurrentStimulator(NeuronGroup * target, AurynFloat initial_current=1.0 );

	/*! Sets the state to add the "current" in every timestep to */
	void set_target_state( string state_name = "mem" );

	/*! Default Destructor */
	virtual ~CurrentStimulator();


	/*! Sets the state that is stimulated with Current input.
	 * This must be a valid state vector name (default = mem) */
	void set_current( NeuronID i, AurynFloat current );

	/*! Implementation of necessary propagate() function. */
	void propagate();

};

#endif /*CURRENTSTIMULATOR_H_*/
