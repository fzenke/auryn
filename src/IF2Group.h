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

#ifndef IF2GROUP_H_
#define IF2GROUP_H_

#include "auryn_definitions.h"
#include "NeuronGroup.h"
#include "System.h"


class IF2Group : public NeuronGroup
{
private:
	auryn_vector_float * t_leak;
	auryn_vector_float * t_exc;
	auryn_vector_float * t_inh;

	auryn_vector_float * nmda_opening;
	AurynFloat scale_ampa,scale_gaba, scale_thr;
	AurynFloat e_rest,e_rev,thr_rest,tau_mem,tau_thr,dthr;
	AurynFloat tau_ampa,tau_gaba,tau_nmda;
	AurynFloat A_ampa,A_nmda;
	AurynFloat e_nmda_onset;
	AurynFloat nmda_slope;
	void init();
	void free();
	void calculate_scale_constants();
	void integrate_membrane();
	void integrate_nonlinear_nmda_synapses();
	void check_thresholds();
public:
	IF2Group( NeuronID size, AurynFloat load = 1.0, NeuronID total = 0 );
	virtual ~IF2Group();
	void set_tau_mem(AurynFloat taum);
	AurynFloat get_tau_mem();
	void set_tau_ampa(AurynFloat tau);
	void set_tau_gaba(AurynFloat tau);
	void set_tau_nmda(AurynFloat tau);
	AurynFloat get_tau_ampa();
	AurynFloat get_tau_gaba();
	AurynFloat get_tau_nmda();
	void set_ampa_nmda_ratio(AurynFloat ratio);
	void clear();
	virtual void evolve();
};

#endif /*IF2GROUP_H_*/

