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

#include "IzhikevichGroup.h"

using namespace auryn;

IzhikevichGroup::IzhikevichGroup(NeuronID size) : NeuronGroup(size)
{
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void IzhikevichGroup::init()
{
	e_rev_gaba = -80e-3;
	tau_ampa = 5e-3;
	tau_gaba = 10e-3;
    r_mem = 1e8;
    c_mem = 200e-12;
	tau_mem = r_mem*c_mem;

	avar = 0.02;   // adaptation variable rate constant
	bvar = 0.2;    // subtreshold adaptation
	cvar = -65e-3; // reset voltage
	dvar = 2.0e-3; // spike triggered adaptation

	e_rest = cvar;
	thr = 30e-3;

	bg_current = get_state_vector("bg_current");
	adaptation_vector = get_state_vector("izhi_adaptation");
	i_exc = get_state_vector("i_exc");
	i_inh = get_state_vector("i_inh");
	temp_vector = get_state_vector("_temp");


	clear();
	calculate_scale_constants();
}

void IzhikevichGroup::clear()
{
	clear_spikes();
   mem->set_all(e_rest);
   g_ampa->set_all(0.);
   g_gaba->set_all(0.);
   bg_current->set_all(0.);
}


IzhikevichGroup::~IzhikevichGroup()
{
	if ( !evolve_locally() ) return;
}

void IzhikevichGroup::check_thresholds()
{
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
    	if ( mem->get(i) > thr ) {
			push_spike(i);
		    mem->set( i, cvar); // reset mem
		    adaptation_vector->add_specific( i, dvar); // increase adapt variable
		} 
	}

}

void IzhikevichGroup::calculate_scale_constants()
{
	scale_ampa =  exp(-dt/tau_ampa) ;
	scale_gaba =  exp(-dt/tau_gaba) ;
}


void IzhikevichGroup::evolve()
{
	// TODO do not hardcode parameters and convert to mV
	
	// compute synaptic conductance based currents
    // excitatory
	i_exc->copy(g_ampa);
	i_exc->scale(-1);
	i_exc->mul(mem);
    
    // inhibitory
	i_inh->diff(mem,e_rev_gaba);
	i_inh->mul(g_gaba);

	// compute izhikevich neuronal dynamics
	temp_vector->copy(mem);
	temp_vector->sqr();
	temp_vector->scale(40);

	temp_vector->saxpy(5,mem);
	temp_vector->add(0.140);
	temp_vector->sub(adaptation_vector);

	// add bg current
	temp_vector->add(bg_current);

	// add synaptic currents
	temp_vector->add(i_exc);
	temp_vector->sub(i_inh);

	mem->saxpy(dt/1e-3,temp_vector); // division by 1e-3 due to rescaling of time from ms -> s

	// update adaptation variable
	temp_vector->copy(mem);
	temp_vector->scale(bvar);
	temp_vector->sub(adaptation_vector);

	adaptation_vector->saxpy(avar*dt/1e-3,temp_vector);

	check_thresholds();

	// decay synaptic conductances
	g_ampa->scale(scale_ampa);
	g_gaba->scale(scale_gaba);
}

void IzhikevichGroup::set_bg_current(NeuronID i, AurynFloat current) {
	if ( localrank(i) )
		auryn_vector_float_set ( bg_current , global2rank(i) , current ) ;
}

void IzhikevichGroup::set_bg_currents(AurynFloat current) {
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) 
		auryn_vector_float_set ( bg_current , i , current ) ;
}

void IzhikevichGroup::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}

void IzhikevichGroup::set_r_mem(AurynFloat rm)
{
	r_mem = rm;
	tau_mem = r_mem*c_mem;
	calculate_scale_constants();
}

void IzhikevichGroup::set_c_mem(AurynFloat cm)
{
	c_mem = cm;
	tau_mem = r_mem*c_mem;
	calculate_scale_constants();
}

AurynFloat IzhikevichGroup::get_bg_current(NeuronID i) {
	if ( localrank(i) )
		return auryn_vector_float_get ( bg_current , global2rank(i) ) ;
	else 
		return 0;
}

std::string IzhikevichGroup::get_output_line(NeuronID i)
{
	std::stringstream oss;
	oss << mem->get(i) << " " << g_ampa->get(i) << " " << g_gaba->get(i) << " " 
		<< bg_current->get(i) <<"\n";
	return oss.str();
}

void IzhikevichGroup::load_input_line(NeuronID i, const char * buf)
{
		float vmem,vampa,vgaba,vbgcur;
		sscanf (buf,"%f %f %f %f",&vmem,&vampa,&vgaba,&vbgcur);
		if ( localrank(i) ) {
			NeuronID trans = global2rank(i);
			mem->set(trans,vmem);
			g_ampa->set(trans,vampa);
			g_gaba->set(trans,vgaba);
			bg_current->set(trans, vbgcur);
		}
}

void IzhikevichGroup::set_tau_ampa(AurynFloat taum)
{
	tau_ampa = taum;
	calculate_scale_constants();
}

AurynFloat IzhikevichGroup::get_tau_ampa()
{
	return tau_ampa;
}

void IzhikevichGroup::set_tau_gaba(AurynFloat taum)
{
	tau_gaba = taum;
	calculate_scale_constants();
}

AurynFloat IzhikevichGroup::get_tau_gaba()
{
	return tau_gaba;
}

void IzhikevichGroup::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
}

void IzhikevichGroup::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
}
