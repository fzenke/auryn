/*
* Copyright 2014-2016 Ankur Sinha and Friedemann Zenke
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

#ifndef ADEXGROUP_H_
#define ADEXGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "System.h"


namespace auryn {

/*! \brief Conductance based Adaptive Exponential neuron model - Brette and Gerstner (2005). 
 *
 * This implements a NeuronGroup of AdEx neurons with default parameters from
 * Brette, R., and Gerstner, W. (2005). Adaptive Exponential Integrate-and-Fire
 * Model as an Effective Description of Neuronal Activity. J Neurophysiol 94,
 * 3637–3642.
 *
 *
 * */
class AdExGroup : public NeuronGroup
{
private:
    AurynFloat e_rest, e_reset, e_rev_gaba, e_rev_ampa,e_thr, g_leak, c_mem, deltat;
    AurynFloat tau_ampa, tau_gaba, tau_mem;
    AurynFloat scale_ampa, scale_gaba, scale_mem, scale_w, scale_current;
    AurynFloat * t_w;
    AurynFloat a; //!< subthreshold adaptation variable in S/g_leak
	AurynFloat b; //!< spike triggered adaptation variable in A/g_leak
	AurynFloat tau_w; //!< adaptation time constant in s
    unsigned short refractory_time;
    auryn_vector_ushort * ref;

    /*! Stores the adaptation current. */
    AurynStateVector * w;

    AurynStateVector * I_exc;
    AurynStateVector * I_inh;
    AurynStateVector * I_leak;
    AurynStateVector * temp;

    AurynFloat * t_g_ampa;
    AurynFloat * t_g_gaba;
    AurynFloat * t_mem;
    unsigned short * t_ref;

    void init();
    void calculate_scale_constants();
    inline void integrate_state();
    inline void check_thresholds();
    virtual std::string get_output_line(NeuronID i);
    virtual void load_input_line(NeuronID i, const char * buf);

	void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version );
	void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version );
public:
    /*! The default constructor of this NeuronGroup */
    AdExGroup(NeuronID size);
    virtual ~AdExGroup();

    /*! Setter for refractory time [s] */
    void set_refractory_period(AurynDouble t);

    /*! \brief Set value of slope factor deltat (default 2mV) */
    void set_delta_t(AurynFloat d);

    /*! \brief Sets the leak conductance (default 30nS) 
	 *
	 * Because these units are used to derive internal numerical values, they need to be set first! */
    void set_g_leak(AurynFloat g);

    /*! \brief Sets the membrane capacitance (default 281pF) 
	 *
	 * Because these units are used to derive internal numerical values, they need to be set first! 
	 * */
    void set_c_mem(AurynFloat cm);

    /*! \brief Set value of a in units S ( default 4nS )
	 *
	 * Internally this is value s converted to natural units of g_leak for numerical stability.
	 * Thus, make sure you set g_leak first!
	 * */
    void set_a(AurynFloat _a);

    /*! \brief Set value of b in units of A ( default 0.0805nA )
	 *
	 * Internally this is value s converted to natural units of g_leak for numerical stability.
	 * Thus, make sure you set g_leak first!
	 * */
    void set_b(AurynFloat _b);

    /*! Set value of V_r (default -70.6mV) */
    void set_e_reset(AurynFloat ereset);

    /*! Set value of E_l (default -70.6mV) */
    void set_e_rest(AurynFloat erest);

    /*! Set value of V_t (default -50.4mV) */
    void set_e_thr(AurynFloat ethr);

    /*! Sets the w time constant (default 144ms) */
    void set_tau_w(AurynFloat tauw);


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

}

#endif /*ADEXGROUP_H_*/

