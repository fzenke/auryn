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

#include "CubaIFGroup.h"

using namespace auryn;

CubaIFGroup::CubaIFGroup(NeuronID size) : NeuronGroup(size)
{
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void CubaIFGroup::calculate_scale_constants()
{
	scale_mem  = dt/tau_mem;
}

void CubaIFGroup::init()
{
	e_rest = -60e-3;
	e_rev = -80e-3;
	thr = -50e-3;
	tau_mem = 20e-3;
	set_refractory_period(5e-3);

	calculate_scale_constants();
	
	ref = auryn_vector_ushort_alloc (get_vector_size()); 
	bg_current = get_state_vector("bg_current");

	t_bg_cur = auryn_vector_float_ptr ( bg_current , 0 ); 
	t_mem = auryn_vector_float_ptr ( mem , 0 ); 
	t_ref = auryn_vector_ushort_ptr ( ref , 0 ); 

	clear();

}

void CubaIFGroup::clear()
{
	clear_spikes();
	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   auryn_vector_float_set (mem, i, e_rest);
	   auryn_vector_ushort_set (ref, i, 0);
	   auryn_vector_float_set (bg_current, i, 0.);
	}
}


CubaIFGroup::~CubaIFGroup()
{
	if ( !evolve_locally() ) return;

	auryn_vector_ushort_free (ref);
}


void CubaIFGroup::evolve()
{


	for (NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
    	if (t_ref[i]==0) {
			const AurynFloat dg_mem = ( (e_rest-t_mem[i]) 
					+ t_bg_cur[i] );
			t_mem[i] += dg_mem*scale_mem;

			if (t_mem[i]>thr) {
				push_spike(i);
				t_mem[i] = e_rest ;
				t_ref[i] += refractory_time ;
			} 
		} else {
			t_ref[i]-- ;
			t_mem[i] = e_rest ;
		}

	}

}

void CubaIFGroup::set_bg_current(NeuronID i, AurynFloat current) {
	if ( localrank(i) )
		auryn_vector_float_set ( bg_current , global2rank(i) , current ) ;
}

void CubaIFGroup::set_all_bg_currents( AurynFloat current ) {
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) 
		auryn_vector_float_set ( bg_current, i, current ) ;
}

void CubaIFGroup::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}

AurynFloat CubaIFGroup::get_bg_current(NeuronID i) {
	if ( localrank(i) )
		return auryn_vector_float_get ( bg_current , global2rank(i) ) ;
	else 
		return 0;
}

std::string CubaIFGroup::get_output_line(NeuronID i)
{
	std::stringstream oss;
	oss << get_mem(i) << " " << auryn_vector_ushort_get (ref, i) << "\n";
	return oss.str();
}

void CubaIFGroup::load_input_line(NeuronID i, const char * buf)
{
		float vmem;
		NeuronID vref;
		sscanf (buf,"%f %u",&vmem,&vref);
		if ( localrank(i) ) {
			NeuronID trans = global2rank(i);
			set_mem(trans,vmem);
			auryn_vector_ushort_set (ref, trans, vref);
		}
}


void CubaIFGroup::set_refractory_period(AurynDouble t)
{
	refractory_time = (unsigned short) (t/dt);
}
