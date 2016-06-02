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

#ifndef LINEARTRACE_H_
#define LINEARTRACE_H_

#include "auryn_definitions.h"
#include "AurynVector.h"

namespace auryn {


/*! \brief Exponential synaptic trace which exactly solves in an event-based manner.
 *
 * Since this trace is normally slower than brut-force computing all traces
 * with forward Euler (EulerTrace), it is disabled by default.
 */

class LinearTrace
{
private:
	NeuronID size;
	AurynFloat * state;
	AurynFloat * explut;
	AurynFloat tau;
	AurynTime tau_auryntime;
	AurynTime zerointerval;
	AurynTime zerotime_auryntime;
	AurynTime * timestamp;
	AurynTime * clock;
	void init(NeuronID n, AurynFloat timeconstant, AurynTime * clk);
	void free();

public:
	LinearTrace(NeuronID n, AurynFloat timeconstant, AurynTime * clk);
	virtual ~LinearTrace();
	void set(NeuronID i , AurynFloat value);
	void setall( AurynFloat value);
	void add(NeuronID i , AurynFloat value);
	void inc(NeuronID i);
	inline void update(NeuronID i);
	void evolve();
	AurynFloat get_tau();
	AurynFloat get(NeuronID i);
	AurynFloat * get_state_ptr();
};

}


#endif /*LINEARTRACE_H_*/

