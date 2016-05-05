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

#include "SIFGroup.h"

using namespace auryn;

SIFGroup::SIFGroup(NeuronID size) : NeuronGroup(size)
{
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void SIFGroup::calculate_scale_constants()
{
	scale_mem  = dt/tau_mem;
	scale_cursyn  = exp(-dt/tau_cursyn);
	scale_ampa = exp(-dt/tau_ampa);
	scale_gaba = exp(-dt/tau_gaba);
}

void SIFGroup::init()
{
	e_rest = 0e-3;
	e_rev = -80e-3;
	thr = .1;
	tau_ampa = 4e-3;
	tau_gaba = 4e-3;
	tau_cursyn = 4e-3;
	tau_mem = 1e-3;
	set_refractory_period(4e-3);

	calculate_scale_constants();
	
	ref = auryn_vector_ushort_alloc (get_vector_size()); 
	g_cursyn = get_state_vector("g_cursyn");
	bg_current = get_state_vector("bg_current");
	inj_current = get_state_vector("inj_current");

	t_g_ampa = auryn_vector_float_ptr ( g_ampa , 0 ); 
	t_g_gaba = auryn_vector_float_ptr ( g_gaba , 0 ); 
	t_g_cursyn = auryn_vector_float_ptr ( g_cursyn , 0 ); 
	t_bg_cur = auryn_vector_float_ptr ( bg_current , 0 ); 
	t_inj_cur = auryn_vector_float_ptr ( inj_current , 0 ); 
	t_mem = auryn_vector_float_ptr ( mem , 0 ); 
	t_ref = auryn_vector_ushort_ptr ( ref , 0 ); 

	clear();

}

void SIFGroup::clear()
{
	clear_spikes();
	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   auryn_vector_ushort_set (ref, i, 0);
	   auryn_vector_float_set (mem, i, e_rest);
	   auryn_vector_float_set (g_ampa, i, 0.);
	   auryn_vector_float_set (g_gaba, i, 0.);
	   auryn_vector_float_set (g_cursyn, i, 0.);
	   auryn_vector_float_set (bg_current, i, 0.);
	   auryn_vector_float_set (inj_current, i, 0.);
	}
}


SIFGroup::~SIFGroup()
{
	if ( !evolve_locally() ) return;

	auryn_vector_ushort_free (ref);
}


void SIFGroup::evolve()
{


	for (NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
    	if (t_ref[i]==0) {
			const AurynFloat dg_mem = ( 
                    + (e_rest-t_mem[i]) 
					- t_g_ampa[i] * t_mem[i]
					- t_g_gaba[i] * (t_mem[i]-e_rev)
					+ t_g_cursyn[i]
					+ t_inj_cur[i]
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

    auryn_vector_float_scale(scale_ampa,g_ampa);
    auryn_vector_float_scale(scale_gaba,g_gaba);
    auryn_vector_float_scale(scale_cursyn,g_cursyn);
}

void SIFGroup::set_bg_current(NeuronID i, AurynFloat current) {
	if ( localrank(i) )
		auryn_vector_float_set ( bg_current , global2rank(i) , current ) ;
}

void SIFGroup::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}

AurynFloat SIFGroup::get_bg_current(NeuronID i) {
	if ( localrank(i) )
		return auryn_vector_float_get ( bg_current , global2rank(i) ) ;
	else 
		return 0;
}

std::string SIFGroup::get_output_line(NeuronID i)
{
	std::stringstream oss;
	oss << mem->get(i) << " " << g_ampa->get(i) << " " << g_gaba->get(i) << " " << ref->get(i) << "\n";
	return oss.str();
}

void SIFGroup::load_input_line(NeuronID i, const char * buf)
{
		float vmem,vampa,vgaba;
		NeuronID vref;
		sscanf (buf,"%f %f %f %u",&vmem,&vampa,&vgaba,&vref);
		if ( localrank(i) ) {
			NeuronID trans = global2rank(i);
			mem->set(trans,vmem);
			g_ampa->set(trans,vampa);
			g_gaba->set(trans,vgaba);
			ref->set( trans, vref);
		}
}

void SIFGroup::set_tau_ampa(AurynFloat taum)
{
	tau_ampa = taum;
	calculate_scale_constants();
}

AurynFloat SIFGroup::get_tau_ampa()
{
	return tau_ampa;
}

void SIFGroup::set_tau_gaba(AurynFloat taum)
{
	tau_gaba = taum;
	calculate_scale_constants();
}

AurynFloat SIFGroup::get_tau_gaba()
{
	return tau_gaba;
}

void SIFGroup::set_refractory_period(AurynDouble t)
{
    double tmp = (unsigned short) (t/dt) - 1;
    if (tmp<0) tmp = 0;
	refractory_time = tmp;
}

void SIFGroup::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
	ar & *ref;
}

void SIFGroup::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
	ar & *ref;
}
