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

#ifndef BINARYSTATEMONITOR_H_
#define BINARYSTATEMONITOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Monitor.h"
#include "System.h"
#include "Connection.h"
#include <fstream>
#include <iomanip>

namespace auryn {

/*! \brief Records from an arbitray state vector of one unit from the source SpikingGroup to a binary file. 
 *
 * Binary state files need to be read with the command line tool aubs or with custom code. */
class BinaryStateMonitor : public Monitor
{
private:
	static const std::string default_extension;

	void write_frame(const AurynTime time, const AurynState value);
	void open_output_file(std::string filename);

protected:
	/*! \brief The source SpikingGroup to record from */
	SpikingGroup * src;

	/*! \brief Target variable */
	AurynState * target_variable;

	/*! \brief Last value (used for compression) */
	AurynState lastval;
	AurynState lastder;

	/*! \brief The source neuron id to record from */
	NeuronID nid;

	/*! \brief The step size (sampling interval) in units of auryn_timestep */
	AurynTime ssize;

	/*! \brief Defines the maximum recording time in AurynTime to save space. */
	AurynTime t_stop;

	/*! \brief Standard initialization */
	void init(string filename, AurynDouble stepsize);
	
public:
	/*! \brief Switch to enable/disable output compression 
	 *
	 * When set to true, Auryn will compute the derivative of two consecutive datapoints of the state
	 * and only a value if the derivative changes. The correct function can then be recovered with linear
	 * interpolation (i.e. plotting in gnuplot with lines will yield the correct output).
	 * When compression is enabled the last value written to file always by one timestep with respect to
	 * the simulation clock.
	 * enable_compression is true by default.
	 */
	bool enable_compression;

	/*! \brief Standard constructor 
	 *
	 * \param source The neuron group to record from
	 * \param id The neuron id in the group to record from 
	 * \param statename The name of the StateVector to record from
	 * \param filename The filename of the file to dump the output to
	 * \param sampling_interval The sampling interval in seconds
	 */
	BinaryStateMonitor(SpikingGroup * source, NeuronID id, string statename, std::string filename="", AurynDouble sampling_interval=auryn_timestep);

	/*! \brief Alternative constructor
	 *
	 * \param state The source state vector
	 * \param filename The filename of the file to dump the output to
	 * \param sampling_interval The sampling interval in seconds
	 */
	BinaryStateMonitor(auryn_vector_float * state, NeuronID id, std::string filename="", AurynDouble sampling_interval=auryn_timestep);

	/*! \brief Trace constructor
	 *
	 * \param trace The source synaptic trace
	 * \param filename The filename of the file to dump the output to
	 * \param sampling_interval The sampling interval in seconds
	 */
	BinaryStateMonitor(Trace * trace, NeuronID id, std::string filename="", AurynDouble sampling_interval=auryn_timestep);

	/*! \brief Sets relative time at which to stop recording 
	 *
	 * The time is given in seconds and interpreted as relative time with 
	 * respect to the current clock value. This features is useful to decrease
	 * IO. The stop time can be set again after calling run to record multiple 
	 * snippets. */
	void record_for(AurynDouble time=10.0);

	/*! \brief Set an absolute time when to stop recording. */
	void set_stop_time(AurynDouble time=10.0);

	virtual ~BinaryStateMonitor();
	void execute();
};

}

#endif /*BINARYSTATEMONITOR_H_*/
