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

#ifndef IAFPSCDELTAGROUP_H_
#define IAFPSCDELTAGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "System.h"

namespace auryn {

/*! \brief Conductance based neuron model with absolute refractoriness as used in Vogels and Abbott 2005.
 */
class IafPscDeltaGroup : public NeuronGroup
{
private:
	AurynFloat scale_mem;

	AurynFloat * t_bg_cur; 
	AurynFloat * t_mem; 

	unsigned short refractory_time;
	auryn_vector_ushort * ref;
	unsigned short * t_ref; 

	void init();
	void calculate_scale_constants();
	inline void integrate_state();
	inline void check_thresholds();
	virtual string get_output_line(NeuronID i);
	virtual void load_input_line(NeuronID i, const char * buf);
public:
	AurynFloat e_rest,e_reset,thr,tau_mem;

	/*! The default constructor of this NeuronGroup */
	IafPscDeltaGroup(NeuronID size);
	virtual ~IafPscDeltaGroup();

	/*! Sets the membrane time constant (default 20ms) */
	void set_tau_mem(AurynFloat taum);

	/*! Sets the refractory time (default is 5ms) */
	void set_tau_ref(AurynFloat tau_ref);

	/*! Resets all neurons to defined and identical initial state. */
	void clear();

	/*! The evolve method internally used by System. */
	void evolve();
};

}

#endif /*IAFPSCDELTAGROUP_H_*/

