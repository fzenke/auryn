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
#include "auryn_global.h"
#include "EulerTrace.h"

namespace auryn {


/*! \brief Exponential synaptic trace which exactly solves in an event-based manner in non-follow scenarios.
 *
 * Since this trace is normally slower than brute-force computing all traces
 * with forward Euler (EulerTrace), it is disabled by default.
 */

class LinearTrace : public EulerTrace
{
private:
	typedef EulerTrace super;
	AurynTime tau_auryntime;
	AurynTime zerointerval;
	AurynTime * timestamp;
	AurynTime * clock;
	void init(NeuronID n, AurynFloat timeconstant);
	void free();

public:
	/* \brief Default constructor */
	LinearTrace(NeuronID n, AurynFloat timeconstant);
	/* \brief Test constructor which allows to init the class without Auryn kernel
	 *
	 * This is only used in tests and should not be used in a simulation. */
	LinearTrace(NeuronID n, AurynFloat timeconstant, AurynTime * clk);
	virtual ~LinearTrace();
	void inc(NeuronID i);
	void add_specific(NeuronID i, AurynState amount);
	AurynFloat get(NeuronID i);
	void update(NeuronID i);
	void evolve();
};

}


#endif /*LINEARTRACE_H_*/

