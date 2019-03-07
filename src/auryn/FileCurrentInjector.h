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

#ifndef FILECURRENTINJECTOR_H_
#define FILECURRENTINJECTOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "Logger.h"
#include "Device.h"
#include "NeuronGroup.h"
#include "CurrentInjector.h"


namespace auryn {

/*! \brief Stimulator class which reads "currents" from a file and applies them to arbitrary neuronal states.
 *
 * Most commonly used to inject "currents" to arbitraty neuronal states. Maintains an internal vector with
 * numbers which are added (times auryn_timestep) in each timestep to the neuronal target vector 
 * (per default that is the membrane voltage and hence the operation corresponds to injecting a current).
 * Note that because of this current units of FileCurrentInjector are in a sense arbitrary because they depend 
 * on the neuron model. The scale of inputs can be with the set_scale runction.
 *
 * Upon construction the FileCurrentInjector accepts a human readable file in
 * two-column format which stores the current time series.  The first column
 * should contain the time in s and the second column the corresponding current
 * values. Missing current values are interpolated linearly.
 * Here is an example for a valid time series file:
 *
 *     0.000000 0.000000
 *     0.050000 0.125333
 *     0.100000 0.248690
 *     0.150000 0.368125
 *     0.200000 0.481754
 *     0.250000 0.587785
 *     0.300000 0.684547
 *
 * 
 */

enum FileCurrentInjectorMode { 
	LIST, 
	ALL,  
};

class FileCurrentInjector : public CurrentInjector
{
private:
	void free();

	/*! \brief compute current current value from sys clock */
	AurynState get_current_current_value();

protected:


	/*! \brief Current time series */
	std::vector<AurynState> * current_time_series;

	/*! \brief The loop grid variable */
	AurynTime loop_grid;


	
public:

	/*! \brief Default Constructor 
	 * @param[target] The target group
	 * @param[time_series_file] The path to the file holding the time series to inject
	 * @param[neuron_state_name] The state to manipulate
	 */
	FileCurrentInjector(NeuronGroup * target, std::string time_series_file, std::string neuron_state_name="mem" );

	/*! \brief Default Destructor */
	virtual ~FileCurrentInjector();

	/*! \brief Loads time series file
	 *
	 * @param[filename] The path and filename of the ASCII two-column file to load.
	 * This functions loads the time series to inject from a human readable
	 * ASCII file in two column format. The first column should be time in s.
	 * The second column should include the current value in arbitrary units
	 * (which may depend on the neuron model and the state variable the
	 * current is injected to. */
	void load_time_series_file(std::string filename);

	/*! \brief Mode of operation 
	 *
	 * Determines whether the current should be injected to a list of neurons or all neurons. */
	FileCurrentInjectorMode mode;

	/*! \brief The array holding the target neuron ids */
	std::vector<NeuronID> * target_neuron_ids;

	/*! \brief Loop switch
	 *
	 * Switches looping on/off.
	 **/
	bool loop;

	/*! \brief Set loop grid in units of s
	 *
	 * Sets the the loop grid variable.  This variable determines whether
	 * consecutive replays in loop mode are aligned to this temporal grid.
	 * This should be larger or equal to the length of the input time series
	 * otherwise it may be truncated.  Per default it is set to 1.0s.
	 **/
	void set_loop_grid(double grid);

	/*! Implementation of necessary propagate() function. */
	void execute();

};

}

#endif /*FILECURRENTINJECTOR_H_*/
