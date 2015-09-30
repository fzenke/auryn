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

#ifndef STATEMONITOR_H_
#define STATEMONITOR_H_

#include "auryn_definitions.h"
#include "Monitor.h"
#include "System.h"
#include "Connection.h"
#include <fstream>
#include <iomanip>

using namespace std;

/*! \brief Records from an arbitray state vector of one unit from the source SpikingGroup to a file.*/
class StateMonitor : protected Monitor
{
protected:
	/*! The source SpikingGroup to record from */
	SpikingGroup * src;

	/*! Target variable */
	AurynState * target_variable;

	/*! The source neuron id to record from */
	NeuronID nid;
	/*! The step size (sampling interval) in units of dt */
	AurynTime ssize;
	/*! Defines the maximum recording time in AurynTime to save space. */
	AurynTime t_stop;
	/*! Standard initialization */
	void init(SpikingGroup * source, NeuronID id, string statename, string filename, AurynTime stepsize);
	
public:
	/*! Standard constructor 
	 * \param source The neuron group to record from
	 * \param id The neuron id in the group to record from 
	 * \param statename The name of the StateVector to record from
	 * \param filename The filename of the file to dump the output to
	 * \param sampling_interval The sampling interval in seconds
	 */
	StateMonitor(SpikingGroup * source, NeuronID id, string statename, string filename, AurynDouble sampling_interval=dt);

	/*! Alternative constructor
	 * \param state The soure state vector
	 * \param filename The filename of the file to dump the output to
	 * \param sampling_interval The sampling interval in seconds
	 */
	StateMonitor(auryn_vector_float * state, NeuronID id, string filename, AurynDouble sampling_interval=dt);

	/*! \brief Sets relative time at which to stop recording 
	 *
	 * The time is given in seconds and interpreted as relative time with 
	 * respect to the current clock value. This features is useful to decrease
	 * IO. The stop time can be set again after calling run to record multiple 
	 * snippets. */
	void record_for(AurynDouble time=10.0);

	/*! \brief Same as record_for(time) */
	void set_stop_time(AurynDouble time=10.0);

	virtual ~StateMonitor();
	void propagate();
};

#endif /*STATEMONITOR_H_*/
