/* 
* Copyright 2015 Neftci Emre and Friedemann Zenke
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

#ifndef SIFGROUP_H_
#define SIFGROUP_H_

#include "auryn_definitions.h"
#include "NeuronGroup.h"
#include "System.h"


/*! \brief Conductance based neuron model with absolute refractoriness used for Neural Sampling
 */
class SIFGroup : public NeuronGroup
{
private:
	auryn_vector_float * bg_current;
	auryn_vector_float * inj_current;
	auryn_vector_ushort * ref;
	unsigned short refractory_time;
	AurynFloat e_rest,e_rev,thr,tau_mem;
	AurynFloat tau_ampa,tau_gaba,tau_cursyn;
	AurynFloat scale_ampa, scale_gaba, scale_mem, scale_cursyn;

	AurynFloat * t_g_cursyn; 
	AurynFloat * t_g_ampa; 
	AurynFloat * t_g_gaba; 
	AurynFloat * t_bg_cur; 
	AurynFloat * t_inj_cur; 
	AurynFloat * t_mem; 
	unsigned short * t_ref; 

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
	SIFGroup(NeuronID size);
	virtual ~SIFGroup();

	/*! Controls the constant current input (per default set so zero) to neuron i */
	void set_bg_current(NeuronID i, AurynFloat current);

	/*! Setter for refractory time [s] */
	void set_refractory_period(AurynDouble t);

	/*! Gets the current background current value for neuron i */
	AurynFloat get_bg_current(NeuronID i);
	/*! Sets the membrane time constant (default 20ms) */
	void set_tau_mem(AurynFloat taum);
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

#endif /*SIFGROUP_H_*/

