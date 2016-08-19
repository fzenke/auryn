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

#ifndef TRACE_H_
#define TRACE_H_

#include "AurynVector.h"
#include "auryn_definitions.h"

namespace auryn {

/*! \brief Abstract base class of synaptic traces
 */
class Trace : public AurynStateVector
{
private:

protected:
	/*! Decay time constant in [s]. */
	AurynFloat tau;

public:
	/*! \brief Default constructor */
	Trace(NeuronID n, AurynFloat timeconstant);

	/*! \brief Default destructor */
	virtual ~Trace();

	/*! \brief Increment given trace by 1.
	 *
	 * \param i index of trace to increment.
	 */
	void inc(NeuronID i);

	/*! \brief Increment given traces by 1.
	 *
	 * \param sc SpikeContainer with all neurons to increment.
	 */
	void inc(SpikeContainer * sc);

	/*! \brief Perform Euler step. */
	virtual void evolve() = 0;

	/*! \brief Set the time constant of the trace */
	virtual void set_timeconstant( AurynFloat timeconstant );

	/*! \brief  Get decay time constant */
	AurynFloat get_tau();

	/*! \brief Get trace value of trace dived by tau
	 *
	 * \param i index of trace to get
	 */ 
	AurynFloat normalized_get(NeuronID i);

	/*! \brief Get pointer to state AurynStateVector for fast processing. */
	AurynStateVector * get_state_ptr();

	/*! \brief Set the target vector for follow operation */
	virtual void set_target( AurynStateVector * target ) = 0;

	/*! \brief Follow other trace */
	virtual void follow() = 0;
};


} // namespace

#endif /*TRACE_H_*/

