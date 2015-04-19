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

#ifndef AEIFGROUP_H_
#define AEIFGROUP_H_

#include "auryn_definitions.h"
#include "NeuronGroup.h"
#include "System.h"


/*! \brief Conductance based Adaptive Exponential neuron model - Brette and Gerstner (2005). Default values are taken from Table 1 (4a)  of Naud, Marcille, Clopath and Gerstner (2008)
 */
class aEIFGroup : public NeuronGroup
{
private:
	auryn_vector_float * bg_current;
	auryn_vector_ushort * ref;
	AurynFloat e_rest, e_reset, e_rev_gaba, e_rev_ampa,e_thr, g_leak, c_mem, delta_t;
	AurynFloat tau_ampa, tau_gaba;
	AurynFloat scale_ampa, scale_gaba, scale_mem, scale_w;
    AurynFloat * t_w;
    AurynFloat a, tau_w, b;

	/*! Stores the adaptation current. */
	auryn_vector_float * w __attribute__((aligned(16)));

	AurynFloat * t_g_ampa;
	AurynFloat * t_g_gaba;
	AurynFloat * t_bg_cur;
	AurynFloat * t_mem;

	void init();
	void calculate_scale_constants();
	inline void integrate_state();
	inline void check_thresholds();
	virtual string get_output_line(NeuronID i);
	virtual void load_input_line(NeuronID i, const char * buf);

	void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version );
	void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version );
public:
	/*! The default constructor of this NeuronGroup */
	aEIFGroup(NeuronID size);
	virtual ~aEIFGroup();

	/*! Controls the constant current input to neuron i (default 500pA) */
	void set_bg_current(NeuronID i, AurynFloat current);

    /*! Set value of slope factor delta_t (default 2mV) */
    void set_delta_t(AurynFloat deltat);
    /*! Set value of a (default 2nS) */
    void set_a(AurynFloat _a);
    /*! Set value of b (default 0nS) */
    void set_b(AurynFloat _b);
    /*! Set value of V_r (default -70mV) */
    void set_e_reset(AurynFloat ereset);
    /*! Set value of E_l (default -70mV) */
    void set_e_rest(AurynFloat erest);
	/*! Sets the w time constant (default 30ms) */
	void set_tau_w(AurynFloat tauw);
	/*! Gets the current background current value for neuron i */
	AurynFloat get_bg_current(NeuronID i);
	/*! Sets the leak conductance (default 10nS) */
	void set_g_leak(AurynFloat g);
	/*! Sets the membrane capacitance (default 200pF) */
	void set_c_mem(AurynFloat cm);
	/*! Sets the exponential time constant for the AMPA channel (default 5ms) */
	void set_tau_ampa(AurynFloat tau);
	/*! Gets the exponential time constant for the AMPA channel */
	AurynFloat get_tau_ampa();
	/*! Sets the exponential time constant for the GABA channel (default 10ms) */
	void set_tau_gaba(AurynFloat tau);
	/*! Gets the exponential time constant for the GABA channel */
	AurynFloat get_tau_gaba();
	/*! Resets all neurons to defined and identical initial state. */
	void clear();
	/*! The evolve method internally used by System. */
	void evolve();
};

#endif /*AEIFGROUP_H_*/

