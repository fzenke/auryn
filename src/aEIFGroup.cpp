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

#include "aEIFGroup.h"

aEIFGroup::aEIFGroup(NeuronID size) : NeuronGroup(size)
{
	sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void aEIFGroup::calculate_scale_constants()
{
	scale_mem  = dt/c_mem;
    scale_w = dt/tau_w;
	scale_ampa = exp(-dt/tau_ampa);
	scale_gaba = exp(-dt/tau_gaba);
}

void aEIFGroup::init()
{
	e_rest = -70e-3;
	e_reset = -58e-3;
    e_rev_ampa = 0;
	e_rev_gaba = -80e-3;
	e_thr = -50e-3;
	tau_ampa = 5e-3;
	tau_gaba = 10e-3;
    tau_w = 30e-3;
    g_leak = 10e-9;
    c_mem = 200e-12;
    delta_t = 2e-3;
    a = 2e-9;
    b = 0;

    w = get_state_vector("w");

	calculate_scale_constants();
	
	ref = auryn_vector_ushort_alloc (get_vector_size());
	bg_current = get_state_vector("bg_current");

	t_g_ampa = auryn_vector_float_ptr ( g_ampa , 0 );
	t_g_gaba = auryn_vector_float_ptr ( g_gaba , 0 );
	t_bg_cur = auryn_vector_float_ptr ( bg_current , 0 );
	t_mem = auryn_vector_float_ptr ( mem , 0 );
	t_w = auryn_vector_float_ptr ( w , 0 );

	clear();

}

void aEIFGroup::clear()
{
	clear_spikes();
	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   auryn_vector_float_set (mem, i, e_rest);
	   auryn_vector_ushort_set (ref, i, 0);
	   auryn_vector_float_set (g_ampa, i, 0.);
	   auryn_vector_float_set (g_gaba, i, 0.);
	   auryn_vector_float_set (bg_current, i, 0.);
	}
}


aEIFGroup::~aEIFGroup()
{
	if ( !evolve_locally() ) return;

	auryn_vector_ushort_free (ref);
}


void aEIFGroup::evolve()
{
	for (NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
        t_w[i] += scale_w * (a * (t_mem[i]-e_rest) - t_w[i]);

        t_mem[i] += scale_mem * (
                g_leak * (
                (e_rest-t_mem[i])
                + delta_t * exp((t_mem[i]-e_thr)/delta_t)
                )
                - t_g_ampa[i] * (t_mem[i]-e_rev_ampa)
                - t_g_gaba[i] * (t_mem[i]-e_rev_gaba)
                + (t_bg_cur[i]-t_w[i]));


        if (t_mem[i]>e_thr) {
            push_spike(i);
            t_mem[i] = e_reset;
            t_w[i] += b;
        }
	}

    auryn_vector_float_scale(scale_ampa,g_ampa);
    auryn_vector_float_scale(scale_gaba,g_gaba);
}

void aEIFGroup::set_bg_current(NeuronID i, AurynFloat current) 
{
	if ( localrank(i) )
		auryn_vector_float_set ( bg_current , global2rank(i) , current ) ;
}

void aEIFGroup::set_tau_w(AurynFloat tauw)
{
	tau_w = tauw;
	calculate_scale_constants();
}

void aEIFGroup::set_b(AurynFloat _b)
{
    b = _b;
}

void aEIFGroup::set_e_reset(AurynFloat ereset)
{
    e_reset = ereset;
}
void aEIFGroup::set_e_rest(AurynFloat erest)
{
    e_rest = erest;
	for (NeuronID i = 0; i < get_rank_size(); i++)
	   auryn_vector_float_set (mem, i, e_rest);
}

void aEIFGroup::set_a(AurynFloat _a)
{
    a = _a;
}

void aEIFGroup::set_delta_t(AurynFloat deltat)
{
    delta_t = deltat;
}

void aEIFGroup::set_g_leak(AurynFloat g)
{
	g_leak = g;
	calculate_scale_constants();
}

void aEIFGroup::set_c_mem(AurynFloat cm)
{
	c_mem = cm;
	calculate_scale_constants();
}

AurynFloat aEIFGroup::get_bg_current(NeuronID i) {
	if ( localrank(i) )
		return auryn_vector_float_get ( bg_current , global2rank(i) ) ;
	else 
		return 0;
}

string aEIFGroup::get_output_line(NeuronID i)
{
	stringstream oss;
	oss << get_mem(i) << " " << get_ampa(i) << " " << get_gaba(i) << " " 
		<< auryn_vector_ushort_get (ref, i) << " " 
		<< auryn_vector_float_get (bg_current, i) <<"\n";
	return oss.str();
}

void aEIFGroup::load_input_line(NeuronID i, const char * buf)
{
		float vmem,vampa,vgaba,vbgcur;
		NeuronID vref;
		sscanf (buf,"%f %f %f %u %f",&vmem,&vampa,&vgaba,&vref,&vbgcur);
		if ( localrank(i) ) {
			NeuronID trans = global2rank(i);
			set_mem(trans,vmem);
			set_ampa(trans,vampa);
			set_gaba(trans,vgaba);
			auryn_vector_ushort_set (ref, trans, vref);
			auryn_vector_float_set (bg_current, trans, vbgcur);
		}
}

void aEIFGroup::set_tau_ampa(AurynFloat taum)
{
	tau_ampa = taum;
	calculate_scale_constants();
}

AurynFloat aEIFGroup::get_tau_ampa()
{
	return tau_ampa;
}

void aEIFGroup::set_tau_gaba(AurynFloat taum)
{
	tau_gaba = taum;
	calculate_scale_constants();
}

AurynFloat aEIFGroup::get_tau_gaba()
{
	return tau_gaba;
}

void aEIFGroup::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
	ar & *ref;
}

void aEIFGroup::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
	ar & *ref;
}
