/* 
* Copyright 2014 Friedemann Zenke
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
*/

#ifndef EULERTRACE_H_
#define EULERTRACE_H_

#include "auryn_definitions.h"
#include <gsl/gsl_vector_float.h>
#include <gsl/gsl_blas.h>

using namespace std;

/*! \brief Solves a set of identical linear differential equations with the Euler method. It is used to implement synaptic traces in most STDP models.
 *
 * This solver simultaneoulsy computes linear traces (mostly to implement synapses) using Euler's method. Another solver is readily available within the Auryn framework. The LinearTrace objects solves the same problem but using the analytic solution. This results in less updates. However - so far it turned out to be inferior in performance to the EulerTrace. 
 */
class EulerTrace
{
private:
	/*! The size of the group. */
	NeuronID size;
	/*! The internal state vector. */
	gsl_vector_float * state;
	/*! The target vector for follow operation. */
	gsl_vector_float * target_ptr;
	/*! Temp update vector for follow operation. */
	gsl_vector_float * update;
	/*! Multiplicative factor to downscale the values in every timestep. */
	AurynFloat scale_const;
	/*! Decay time constant in [s]. */
	AurynFloat tau;

	void init(NeuronID n, AurynFloat timeconstant);
	void free();

public:
	/*! Default constructor */
	EulerTrace(NeuronID n, AurynFloat timeconstant);
	/*! Default destructor */
	virtual ~EulerTrace();
	/*! Set value of a single trace.
	 * \param i Id of value to change.
	 * \param value The actual value to set the trace to.*/
	void set(NeuronID i , AurynFloat value);
	/*! Set all traces to same value */
	void setall( AurynFloat value);
	/*! Add gsl_vector_float to state vector
	 * \param values gsl_vector_float to add
	 */
	void add(gsl_vector_float * values);
	/*! Add designated value to single trace in the group.
	 * \param i index of trace to change
	 * \param value value to add to the trace
	 */
	void add(NeuronID i , AurynFloat value);
	/*! Increment given trace by 1.
	 * \param i index of trace to increment.
	 */
	void inc(NeuronID i);
	/*! Perform Euler step. */
	void evolve();

	/*! Clip trace values (0,val) . */
	void clip(AurynState val=1.0);
	
	/*! Set the time constant of the trace */
	void set_timeconstant( AurynFloat timeconstant );

	/*! set the target vector for follow operation */
	void set_target( gsl_vector_float * target );

	/*! set the target vector for follow operation */
	void set_target( EulerTrace * target );

	/*! Perform Euler step but follow target vector instead of zero-decay */
	void follow();
	/*! Get decay time constant */
	AurynFloat get_tau();
	/*! Get trace value of trace 
	 * \param i index of trace to get
	 */ 
	AurynFloat get(NeuronID i);
	/*! Get trace value of trace dived by tau
	 * \param i index of trace to get
	 */ 
	AurynFloat normalized_get(NeuronID i);
	/*! Get pointer to state gsl_vector_float for fast processing within the GSL vector framekwork. */
	gsl_vector_float * get_state_ptr();
};



#endif /*EULERTRACE_H_*/

