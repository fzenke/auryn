/*
* Copyright 2014-2018 Ankur Sinha and Friedemann Zenke
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

#include "AdExGroupNB.h"

using namespace auryn;

AdExGroupNB::AdExGroupNB(NeuronID size) : NeuronGroup(size)
{
    auryn::sys->register_spiking_group(this);
    if ( evolve_locally() ) init();
}

void AdExGroupNB::calculate_scale_constants()
{
    scale_mem  = auryn_timestep/tau_mem;
    scale_w    = auryn_timestep/tau_w;
    scale_ampa = exp(-auryn_timestep/tau_ampa);
    scale_nmda = exp(-auryn_timestep/tau_nmda);
    scale_gaba = exp(-auryn_timestep/tau_gaba);
}

void AdExGroupNB::init()
{
    e_rest = -70.6e-3; // resting potential
    e_reset = e_rest; // reset voltage
    e_thr = -50.4e-3; // V_t spike threshold
    g_leak = 30e-9; // leak conductance
    tau_w = 144e-3; // adaptation time constant
    c_mem = 281e-12; // membrane capacitance
    tau_mem = c_mem/g_leak;
    deltat = 2e-3; // slope factor
    set_a(4e-9); // subthreshold adaptation variable in units of Siemens
    set_b(0.0805e-9); // spike triggered adaptation variable in units of nA

    // conductance based synaptic parameters
    tau_ampa = 5e-3;
    tau_nmda = 100e-3;
    tau_gaba = 10e-3; 
    e_rev_ampa = 0; // Uses same for e_rev_nmda
    e_rev_gaba = -80e-3;

    set_refractory_period(0);

	// init adaptation current variable "w"
    w = get_state_vector("w");

    calculate_scale_constants();
    ref = auryn_vector_ushort_alloc (get_vector_size());

    t_g_ampa = auryn_vector_float_ptr ( g_ampa , 0 );
    t_g_nmda = auryn_vector_float_ptr ( g_nmda , 0 );
    t_g_gaba = auryn_vector_float_ptr ( g_gaba , 0 );
    t_mem = auryn_vector_float_ptr ( mem , 0 );
    t_w = auryn_vector_float_ptr ( w , 0 );
    t_ref = auryn_vector_ushort_ptr ( ref , 0 );

    I_leak = get_state_vector("I_leak");
    I_exc_ampa  = get_state_vector("I_exc_ampa");
    I_exc_nmda  = get_state_vector("I_exc_nmda");
    I_inh  = get_state_vector("I_inh");
    temp   = get_state_vector("_temp");

    clear();

}

void AdExGroupNB::clear()
{
    clear_spikes();

    mem->set_all(e_rest);
    g_ampa->set_zero();
    g_nmda->set_zero();
    g_gaba->set_zero();
    w->set_zero();
    ref->set_zero();
}


AdExGroupNB::~AdExGroupNB()
{
    if ( !evolve_locally() ) return;

    auryn_vector_ushort_free (ref);
}

void AdExGroupNB::evolve()
{
	// Compute
	//     t_mem[i] += scale_mem * (
	//             e_rest-t_mem[i]
	//             + deltat * exp((t_mem[i]-e_thr)/deltat)
	//             - t_g_ampa[i] * (t_mem[i]-e_rev_ampa)
	//             - t_g_nmda[i] * (t_mem[i]-e_rev_ampa) // e_rev_nmda == e_rev_ampa
	//             - t_g_gaba[i] * (t_mem[i]-e_rev_gaba) 
	//             -t_w[i] ) ;
	// as vectorized code

	// Compute currents
	I_leak->diff(e_rest,mem);

	I_exc_ampa->diff(e_rev_ampa, mem);
	I_exc_ampa->mul(g_ampa);

	I_exc_nmda->diff(e_rev_ampa, mem);
	I_exc_nmda->mul(g_nmda);
	
	I_inh->diff(e_rev_gaba, mem);
	I_inh->mul(g_gaba);

	// compute spike generating current 
	temp->diff(mem,e_thr);
	temp->scale(1.0/deltat);
	temp->fast_exp();
	temp->scale(deltat);

	// sum up all the currents
	temp->add(I_leak);
	temp->add(I_exc_ampa);
	temp->add(I_exc_nmda);
	temp->add(I_inh);
	temp->sub(w); // adaptation current

	// Euler update membrane
	mem->saxpy(scale_mem, temp);
	mem->clip(e_rev_gaba, 20e-3); // needs to be larger than 0.0 

	// check thresholds
    for (NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
        if (t_ref[i]==0) {
            if (t_mem[i]>0.0) {
                push_spike(i);
                t_mem[i] = e_reset;
                t_w[i] += b;
                t_ref[i] += refractory_time ;
            }
        } else {
            t_ref[i]-- ;
            t_mem[i] = e_rest ;
        }
    }

	// computes
	// dw = scale_w * (a * (t_mem[i]-e_rest) - t_w[i]);
	// in vector lingo
	temp->diff(mem,e_rest);
	temp->scale(a);
	temp->sub(w);
	// Euler upgrade adaptation variable
	w->saxpy(scale_w,temp);


    g_ampa->scale(scale_ampa);
    g_nmda->scale(scale_nmda);
    g_gaba->scale(scale_gaba);
}

void AdExGroupNB::set_tau_w(AurynFloat tauw)
{
    tau_w = tauw;
    calculate_scale_constants();
}

void AdExGroupNB::set_e_reset(AurynFloat ereset)
{
    e_reset = ereset;
}

void AdExGroupNB::set_e_thr(AurynFloat ethr)
{
    e_thr = ethr;
}
void AdExGroupNB::set_e_rest(AurynFloat erest)
{
    e_rest = erest;
    for (NeuronID i = 0; i < get_rank_size(); i++)
        auryn_vector_float_set (mem, i, e_rest);
}

void AdExGroupNB::set_e_rev_ampa(AurynFloat erev) { // ALa added
	e_rev_ampa = erev;
}

void AdExGroupNB::set_e_rev_gaba(AurynFloat erev) { // ALa added
	e_rev_gaba = erev;
}

void AdExGroupNB::set_a(AurynFloat _a)
{
    a = _a/g_leak;
}

void AdExGroupNB::set_b(AurynFloat _b)
{
    b = _b/g_leak;
}

void AdExGroupNB::set_delta_t(AurynFloat d)
{
    deltat = d/g_leak;
}

void AdExGroupNB::set_g_leak(AurynFloat g)
{
    g_leak = g;
    tau_mem = c_mem/g_leak;
    calculate_scale_constants();
}

void AdExGroupNB::set_c_mem(AurynFloat cm)
{
    c_mem = cm;
    tau_mem = c_mem/g_leak;
    calculate_scale_constants();
}


std::string AdExGroupNB::get_output_line(NeuronID i)
{
    std::stringstream oss;
    oss << mem->get(i) << " " 
		<< g_ampa->get(i) << " " 
		<< g_nmda->get(i) << " " 
		<< g_gaba->get(i) << " "
        << ref->get(i) <<"\n";
    return oss.str();
}

void AdExGroupNB::load_input_line(NeuronID i, const char * buf)
{
    float vmem,vampa,vnmda,vgaba;
        NeuronID vref;
        sscanf (buf,"%f %f %f %f %u",&vmem,&vampa,&vnmda,&vgaba,&vref);
        if ( localrank(i) ) {
            NeuronID trans = global2rank(i);
            mem->set(trans,vmem);
            g_ampa->set(trans,vampa);
            g_nmda->set(trans,vnmda);
            g_gaba->set(trans,vgaba);
            ref->set( trans, vref);
        }
}

void AdExGroupNB::set_tau_ampa(AurynFloat taum)
{
    tau_ampa = taum;
    calculate_scale_constants();
}

AurynFloat AdExGroupNB::get_tau_ampa()
{
    return tau_ampa;
}

void AdExGroupNB::set_tau_nmda(AurynFloat taum)
{
    tau_nmda = taum;
    calculate_scale_constants();
}

AurynFloat AdExGroupNB::get_tau_nmda()
{
    return tau_nmda;
}

void AdExGroupNB::set_tau_gaba(AurynFloat taum)
{
    tau_gaba = taum;
    calculate_scale_constants();
}

AurynFloat AdExGroupNB::get_tau_gaba()
{
    return tau_gaba;
}

void AdExGroupNB::set_refractory_period(AurynDouble t)
{
    refractory_time = (unsigned short) (t/auryn_timestep);
}

void AdExGroupNB::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version )
{
    SpikingGroup::virtual_serialize(ar,version);
    ar & *ref;
}

void AdExGroupNB::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version )
{
    SpikingGroup::virtual_serialize(ar,version);
    ar & *ref;
}
