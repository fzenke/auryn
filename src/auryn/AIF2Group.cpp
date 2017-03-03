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

#include "AIF2Group.h"

using namespace auryn;


AIF2Group::AIF2Group( NeuronID size, NodeDistributionMode distmode ) : AIFGroup(size, distmode)
{
	// auryn::sys->register_spiking_group(this); // already registered in AIFGroup
	if ( evolve_locally() ) init();
}

void AIF2Group::calculate_scale_constants()
{
	AIFGroup::calculate_scale_constants();
	scale_adapt2 = exp(-auryn_timestep/tau_adapt2);
}

void AIF2Group::init()
{
	tau_adapt2 = 20.0;
	dg_adapt2  = 0.002;

	calculate_scale_constants();
	g_adapt2 = get_state_vector ("g_adapt2");

	clear();
}

void AIF2Group::clear()
{
	AIFGroup::clear();
	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   auryn_vector_float_set (g_adapt2, i, 0.);
	 }
}


void AIF2Group::random_adapt(AurynState mean, AurynState sigma)
{
	boost::mt19937 ng_gen(42); // produces same series every time 
	boost::normal_distribution<> dist((double)mean, (double)sigma);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(ng_gen, dist);
	AurynState rv;

	for ( AurynLong i = 0 ; i<get_rank_size() ; ++i ) {
		rv = die();
		if ( rv>0 ) 
			g_adapt1->set( i, rv ); 
		rv = die();
		if ( rv>0 ) 
			g_adapt2->set( i, rv ); 
	}

	init_state();
}

void AIF2Group::free()
{
}

AIF2Group::~AIF2Group()
{
	if ( evolve_locally() ) free();
}

void AIF2Group::integrate_linear_nmda_synapses()
{
	// decay of ampa and gaba channel, i.e. multiply by exp(-auryn_timestep/tau)
    auryn_vector_float_scale(scale_ampa,g_ampa);
    auryn_vector_float_scale(scale_gaba,g_gaba);
    auryn_vector_float_scale(scale_adapt1,g_adapt1);
    auryn_vector_float_scale(scale_adapt2,g_adapt2);

    // compute dg_nmda = (g_ampa-g_nmda)*auryn_timestep/tau_nmda and add to g_nmda
	AurynFloat mul_nmda = auryn_timestep/tau_nmda;
    auryn_vector_float_saxpy(mul_nmda,g_ampa,g_nmda);
	auryn_vector_float_saxpy(-mul_nmda,g_nmda,g_nmda);

    // excitatory
    auryn_vector_float_copy(g_ampa,t_exc);
    auryn_vector_float_scale(-A_ampa,t_exc);
    auryn_vector_float_saxpy(-A_nmda,g_nmda,t_exc);
    auryn_vector_float_mul(t_exc,mem);
    
    // inhibitory
    auryn_vector_float_copy(g_gaba,t_leak); // using t_leak as temp here
    auryn_vector_float_saxpy(1,g_adapt1,t_leak);
    auryn_vector_float_saxpy(1,g_adapt2,t_leak);
    auryn_vector_float_copy(mem,t_inh);
    auryn_vector_float_add_constant(t_inh,-e_rev);
    auryn_vector_float_mul(t_inh,t_leak);
}


void AIF2Group::check_thresholds()
{
	auryn_vector_float_clip( mem, e_rev );

	AurynState * thr_ptr = thr->data;
	for ( AurynState * i = mem->data ; i != mem->data+get_rank_size() ; ++i ) { // it's important to use rank_size here otherwise there might be spikes from units that do not exist
    	if ( *i > ( thr_rest + *thr_ptr ) ) {
			NeuronID unit = i-mem->data;
			push_spike(unit);
		    mem->set( unit, e_rest); // reset
	        thr->set( unit, dthr); //refractory
			g_adapt1->add_specific( unit, dg_adapt1);
			g_adapt2->add_specific( unit, dg_adapt2);
		} 
		thr_ptr++;
	}

}

void AIF2Group::evolve()
{
	integrate_linear_nmda_synapses();
	integrate_membrane();
	check_thresholds();
}



