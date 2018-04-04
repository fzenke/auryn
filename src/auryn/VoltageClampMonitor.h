/* 
* Copyright 2014-2018 Friedemann Zenke
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

#ifndef VOLTAGECLAMPMONITOR_H_
#define VOLTAGECLAMPMONITOR_H_

#define VOLTAGEMONITOR_PASTED_SPIKE_HEIGHT 20e-3

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Monitor.h"
#include "System.h"
#include "Connection.h"
#include <fstream>
#include <iomanip>

namespace auryn {

/*! \brief Implements a voltage clamp for one neuron and records the clamp current 
 *
 * The Monitor puts a single cell in voltage clamp and records the membrane current required ot keep the clamp. 
 * */
class VoltageClampMonitor : public Monitor
{
private:
	/*! \brief Global neuron id to record from */
	NeuronID gid;

protected:
	/*! \brief The source neuron group to record from */
	NeuronGroup * src;

	/*! \brief The source neuron id to record from */
	NeuronID nid;

	/*! \brief Defines the maximum recording time in AurynTime to save space. */
	AurynTime t_stop;

	/*! \brief Standard initialization */
	void init(NeuronGroup * source, NeuronID id, string filename);
	
public:
	/*! \brief Clamp active */
	bool clamp_enabled;

	/*! \brief The clamping voltage */
	AurynState clamping_voltage;

	/*! \brief Sets relative time at which to stop recording 
	 *
	 * The time is given in seconds and interpreted as relative time with 
	 * respect to the current clock value. This features is useful to decrease
	 * IO. The stop time can be set again after calling run to record multiple 
	 * snippets. */
	void record_for(AurynDouble time=10.0);

	/*! \brief Same as record for(time) */
	void set_stop_time(AurynDouble time=10.0);

	/*! \brief Default constructor
	 *
	 * \param source The group to record and clamp
	 * \param id The neuron id to operate on
	 * \param filename The filename to write the clamp current to
	 * */
	VoltageClampMonitor(NeuronGroup * source, NeuronID id, string filename);

	/*! \brief The default destructor */
	virtual ~VoltageClampMonitor();
	void execute();
};

}

#endif /*VOLTAGECLAMPMONITOR_H_*/
