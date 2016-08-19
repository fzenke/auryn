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

#ifndef EULERTRACE_H_
#define EULERTRACE_H_

#include "Trace.h"
#include "auryn_definitions.h"

namespace auryn {

/*! \brief Solves a set of identical linear differential equations with the
 * Euler method. It is used to implement synaptic traces in most STDP models.
 *
 * This solver simultaneoulsy computes linear traces (mostly to implement
 * synapses) using Euler's method. Another solver is readily available within
 * the Auryn framework. The LinearTrace objects solves the same problem but
 * using the analytic solution. This results in less updates. However - so far
 * it turned out to be inferior in performance to the EulerTrace. 
 */
class EulerTrace : public Trace
{
private:
	typedef Trace super;
	/*! The target vector for follow operation. */
	AurynStateVector * target_ptr;

	/*! Temp update vector for follow operation. */
	AurynStateVector * temp;

	/*! Multiplicative factor to downscale the values in every timestep. */
	AurynFloat scale_const;

	/*! \brief precomputed euler upgrade step size. */
	AurynFloat mul_follow;

	void init(NeuronID n, AurynFloat timeconstant);
	void free();

	/*! \brief Checks if argument is larger than size and throws and exception if so 
	 *
	 * Check only enabled if NDEBUG is not defined.*/
	void check_size(NeuronID x)
	{
#ifndef NDEBUG
		if ( x >= size ) {
			throw AurynVectorDimensionalityException();
		}
#endif 
	};

public:
	/*! Default constructor */
	EulerTrace(NeuronID n, AurynFloat timeconstant);
	/*! Default destructor */
	virtual ~EulerTrace();

	/*! Perform Euler step. */
	void evolve();

	/*! Set the time constant of the trace */
	void set_timeconstant( AurynFloat timeconstant );

	/*! set the target vector for follow operation */
	void set_target( AurynStateVector * target );

	/*! Perform Euler step but follow target vector instead of zero-decay */
	void follow();
};

} // namespace

#endif /*EULERTRACE_H_*/

