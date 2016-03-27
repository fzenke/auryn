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

#include "IafPscDeltaGroup.h"

using namespace auryn;

IafPscDeltaGroup::IafPscDeltaGroup(NeuronID size) : NeuronGroup(size)
{
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void IafPscDeltaGroup::calculate_scale_constants()
{
	scale_mem  = dt/tau_mem;
}

void IafPscDeltaGroup::init()
{
	e_rest = -60e-3;
	e_reset = -60e-3;
	thr = -50e-3;
	tau_mem = 20e-3;

	set_tau_ref(2e-3);


	calculate_scale_constants();
	
	t_mem = auryn_vector_float_ptr ( mem , 0 ); 
	ref = auryn_vector_ushort_alloc (get_vector_size()); 
	t_ref = auryn_vector_ushort_ptr ( ref , 0 ); 

	clear();

}

void IafPscDeltaGroup::set_tau_ref(AurynFloat tau_ref)
{
	refractory_time = (unsigned short) (tau_ref/dt);
}

void IafPscDeltaGroup::clear()
{
	clear_spikes();
	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   auryn_vector_float_set (mem, i, e_rest);
	   auryn_vector_ushort_set (ref, i, 0);
	}
}


IafPscDeltaGroup::~IafPscDeltaGroup()
{
	if ( !evolve_locally() ) return;

	auryn_vector_ushort_free (ref);
}


void IafPscDeltaGroup::evolve()
{
	for (NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
    	if (t_ref[i]==0) {
			if (t_mem[i]>thr) {
				push_spike(i);
				t_mem[i] = e_reset ;
				t_ref[i] += refractory_time ;
			} else {
				AurynDouble dg_mem = ( e_rest-t_mem[i] );
				t_mem[i] += dg_mem*scale_mem;
			}
		} else {
			t_mem[i] = e_reset ;
			t_ref[i]-- ;
		}

	}
}


void IafPscDeltaGroup::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}


std::string IafPscDeltaGroup::get_output_line(NeuronID i)
{
	std::stringstream oss;
	oss << get_mem(i) << " " << auryn_vector_ushort_get (ref, i) << "\n";
	return oss.str();
}

void IafPscDeltaGroup::load_input_line(NeuronID i, const char * buf)
{
	float vmem,vampa,vgaba;
	NeuronID vref;
	sscanf (buf,"%f %f %f %u",&vmem,&vampa,&vgaba,&vref);
	if ( localrank(i) ) {
		NeuronID trans = global2rank(i);
		set_mem(trans,vmem);
		auryn_vector_ushort_set (ref, trans, vref);
	}
}
