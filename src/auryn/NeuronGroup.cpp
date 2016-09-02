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

#include "NeuronGroup.h"

using namespace auryn;

NeuronGroup::NeuronGroup(NeuronID n, double loadmultiplier, NeuronID total ) : SpikingGroup(n, loadmultiplier, total )
{
	if ( evolve_locally() ) init();
}

void NeuronGroup::init()
{  		group_name = "NeuronGroup";

		// std::stringstream oss;
		// oss << description_std::string << " init";
		// auryn::logger->msg(oss.str(),VERBOSE);

		mem = get_state_vector("mem");
		thr = get_state_vector("thr");
		g_ampa = get_state_vector("g_ampa");
		g_gaba = get_state_vector("g_gaba");
		g_nmda = get_state_vector("g_nmda");

		default_exc_target_state = g_ampa;
		default_inh_target_state = g_gaba;

#ifndef CODE_ALIGNED_SSE_INSTRUCTIONS
		// checking via default if those arrays are aligned
		if ( auryn_AlignOffset( mem->size, mem->data, sizeof(float), 16) 
				|| auryn_AlignOffset( thr->size, thr->data, sizeof(float), 16) 
				|| auryn_AlignOffset( g_ampa->size, g_ampa->data, sizeof(float), 16) 
				|| auryn_AlignOffset( g_nmda->size, g_nmda->data, sizeof(float), 16) 
				|| auryn_AlignOffset( g_gaba->size, g_gaba->data, sizeof(float), 16)  
		   ) 
			throw AurynMemoryAlignmentException();
#endif
}


void NeuronGroup::free()
{
}


NeuronGroup::~NeuronGroup()
{
	if ( evolve_locally() ) free();
}


void NeuronGroup::random_mem(AurynState mean, AurynState sigma)
{
	randomize_state_vector_gauss("mem",mean,sigma,42);
	init_state();
}

void NeuronGroup::safe_tadd(NeuronID id, AurynWeight amount, TransmitterType t)
{
	if (localrank(id))
		tadd(id, amount, t);
}

void NeuronGroup::init_state()
{

}

void NeuronGroup::tadd(NeuronID id, AurynWeight amount, TransmitterType t)
{
	NeuronID localid = global2rank(id);
	switch ( t ) {
		case GABA:
			g_gaba->add_specific(localid,amount);
			break;
		case MEM:
			mem->add_specific(localid,amount);
			break;
		case NMDA:
			g_nmda->add_specific(localid,amount);
			break;
		case GLUT:
		case AMPA:
		default:
			g_ampa->add_specific(localid,amount);
	}
}


void NeuronGroup::tadd(AurynStateVector * state, NeuronID id, AurynWeight amount)
{
	NeuronID localid = global2rank(id);
	state->add_specific( localid, amount);
}

void NeuronGroup::set_state(std::string name, NeuronID i, AurynState val)
{
	AurynStateVector * tmp = find_state_vector(name);
	if (tmp) { tmp->set(i,val); }
	else { logger->warning("State not found."); }
}

void NeuronGroup::set_state(std::string name, AurynState val)
{
	AurynStateVector * tmp = find_state_vector(name);
	if (tmp) tmp->set_all(val);
	else { logger->warning("State not found."); }
}

AurynStateVector * NeuronGroup::get_default_exc_target()
{
	return default_exc_target_state;
}

AurynStateVector * NeuronGroup::get_default_inh_target()
{
	return default_inh_target_state;
}

