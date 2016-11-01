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

#include "IafPscExpGroup.h"

using namespace auryn;

IafPscExpGroup::IafPscExpGroup(NeuronID size) : NeuronGroup(size)
{
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void IafPscExpGroup::calculate_scale_constants()
{
	scale_mem  = auryn_timestep/tau_mem;
	scale_syn  = exp(-auryn_timestep/tau_syn);
}

void IafPscExpGroup::init()
{
	e_rest = -60e-3;
	thr = -50e-3;
	tau_syn = 5e-3;
    r_mem = 1e8;
    c_mem = 200e-12;
	tau_mem = r_mem*c_mem;
	set_refractory_period(5e-3);

	calculate_scale_constants();
	
	bg_current = get_state_vector("bg_current");
	temp = get_state_vector("_temp");
	ref = new AurynVector< unsigned short > (get_vector_size()); 

	syn_current = get_state_vector("syn_current");
	default_exc_target_state = syn_current;
	default_inh_target_state = syn_current;


	t_mem = mem->ptr( ); 
	t_ref = ref->ptr( ); 

	clear();

}

void IafPscExpGroup::clear()
{
	clear_spikes();
	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   mem->set( i, e_rest);
	   ref->set( i, 0);
	   g_ampa->set( i, 0.);
	   g_gaba->set( i, 0.);
	   bg_current->set( i, 0.);
	}
}


IafPscExpGroup::~IafPscExpGroup()
{
	if ( !evolve_locally() ) return;

	delete ref;
}


void IafPscExpGroup::evolve()
{

	// integrate membrane

    // compute current
    temp->diff(e_rest, mem); // leak current
	temp->add(bg_current); // bg current
	temp->add(syn_current); // syn_current

    // membrane dynamics
    mem->saxpy(scale_mem,temp);

	// hard refractory time (clamped to zero)
	for (NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
    	if (t_ref[i]==0) {
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

	syn_current->scale(scale_syn);
}

void IafPscExpGroup::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}

void IafPscExpGroup::set_r_mem(AurynFloat rm)
{
	r_mem = rm;
	tau_mem = r_mem*c_mem;
	calculate_scale_constants();
}

void IafPscExpGroup::set_c_mem(AurynFloat cm)
{
	c_mem = cm;
	tau_mem = r_mem*c_mem;
	calculate_scale_constants();
}

std::string IafPscExpGroup::get_output_line(NeuronID i)
{
	std::stringstream oss;
	oss << mem->get(i) << " " << g_ampa->get(i) << " " << g_gaba->get(i) << " " 
		<< ref->get(i) << " " 
		<< bg_current->get(i) <<"\n";
	return oss.str();
}

void IafPscExpGroup::load_input_line(NeuronID i, const char * buf)
{
		float vmem,vampa,vgaba,vbgcur;
		NeuronID vref;
		sscanf (buf,"%f %f %f %u %f",&vmem,&vampa,&vgaba,&vref,&vbgcur);
		if ( localrank(i) ) {
			NeuronID trans = global2rank(i);
			mem->set(trans,vmem);
			g_ampa->set(trans,vampa);
			g_gaba->set(trans,vgaba);
			ref->set(trans, vref);
			bg_current->set(trans, vbgcur);
		}
}

void IafPscExpGroup::set_tau_syn(AurynFloat tau)
{
	tau_syn = tau;
	calculate_scale_constants();
}

void IafPscExpGroup::set_refractory_period(AurynDouble t)
{
	refractory_time = (unsigned short) (t/auryn_timestep);
}

void IafPscExpGroup::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
	ar & *ref;
}

void IafPscExpGroup::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
	ar & *ref;
}
