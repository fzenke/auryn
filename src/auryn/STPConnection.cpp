/* 
* Copyright 2014-2023 Friedemann Zenke
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

#include "STPConnection.h"

using namespace auryn;

void STPConnection::init() 
{
	if ( src->get_rank_size() > 0 ) {
		// init of STP stuff
		tau_d = 0.2;
		tau_f = 1.0;
		Ujump = 0.01;
		state_x = auryn_vector_float_alloc( src->get_rank_size() );
		state_u = auryn_vector_float_alloc( src->get_rank_size() );
		state_temp = auryn_vector_float_alloc( src->get_rank_size() );
		clear();
	}

	// Registering the right amount of spike attributes
	// this line is very important finding bugs due to 
	// this being wrong or missing is hard 
	add_number_of_spike_attributes(1);

}


STPConnection::STPConnection(const char * filename) 
: SparseConnection(filename)
{
	init();
}

STPConnection::STPConnection(SpikingGroup * source, NeuronGroup * destination, 
		TransmitterType transmitter) 
: SparseConnection(source, destination, transmitter)
{
}

STPConnection::STPConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename , 
		TransmitterType transmitter) 
: SparseConnection(source, destination, filename, transmitter)
{
	init();
}


STPConnection::STPConnection(NeuronID rows, NeuronID cols) 
: SparseConnection(rows,cols)
{
	init();
}

STPConnection::STPConnection( SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		TransmitterType transmitter, std::string name) 
: SparseConnection(source,destination,weight,sparseness,transmitter, name)
{
	init();
}

void STPConnection::free()
{
	if ( src->get_rank_size() > 0 ) {
		auryn_vector_float_free (state_x);
		auryn_vector_float_free (state_u);
		auryn_vector_float_free (state_temp);
	}
}



STPConnection::~STPConnection()
{
	free();
}

void STPConnection::push_attributes()
{
	// need to push one attribute for each spike
	SpikeContainer * spikes = src->get_spikes_immediate();
	for (SpikeContainer::const_iterator spike = spikes->begin() ;
			spike != spikes->end() ; ++spike ) {
		// dynamics 
		NeuronID spk = src->global2rank(*spike);
		double x = auryn_vector_float_get( state_x, spk );
		double u = auryn_vector_float_get( state_u, spk );
		auryn_vector_float_set( state_x, spk, x-u*x );
		auryn_vector_float_set( state_u, spk, u+Ujump*(1-u) );

		// TODO spike translation or introduce local_spikes 
		// function in SpikingGroup and implement this there ... (better option)
		src->push_attribute( x*u ); 

	}

	// If we had two spike attributes in this connection we push 
	// the second attribute for each spike here:
	//
	// SpikeContainer * spikes = src->get_spikes_immediate();
	// for (SpikeContainer::const_iterator spike = spikes->begin() ;
	// 		spike != spikes->end() ; ++spike ) {
	// 	AurynFloat other_attribute = foo+bar;
	// 	src->push_attribute( other_attribute ); 
	// }
}

void STPConnection::evolve()
{
	if ( src->evolve_locally() ) {
		// dynamics of x
		auryn_vector_float_set_all( state_temp, 1);
		auryn_vector_float_saxpy(-1,state_x,state_temp);
		auryn_vector_float_saxpy(auryn_timestep/tau_d,state_temp,state_x);

		// dynamics of u
		auryn_vector_float_set_all( state_temp, Ujump);
		auryn_vector_float_saxpy(-1,state_u,state_temp);
		auryn_vector_float_saxpy(auryn_timestep/tau_f,state_temp,state_u);
	}
}

void STPConnection::propagate()
{
	if ( src->evolve_locally()) {
		push_attributes(); // stuffs all attributes into the SpikeDelays for sync
	}

	if ( dst->evolve_locally() ) { // necessary 
		NeuronID * ind = w->get_row_begin(0); // first element of index array
		AurynWeight * data = w->get_data_begin(); // first element of data array

		// loop over spikes
		for (NeuronID i = 0 ; i < src->get_spikes()->size() ; ++i ) {
			// get spike at pos i in SpikeContainer
			NeuronID spike = src->get_spikes()->at(i);

			// extract spike attribute from attribute stack;
			AurynFloat attribute = get_spike_attribute(i);

			// loop over postsynaptic targets
			for (NeuronID * c = w->get_row_begin(spike) ; 
					c != w->get_row_end(spike) ; 
					++c ) {
				AurynWeight value = data[c-ind] * attribute; 
				transmit( *c , value );
			}
		}
	}
}

void STPConnection::set_tau_f(AurynFloat tauf) {
	tau_f = tauf;
}

void STPConnection::set_tau_d(AurynFloat taud) {
	tau_d = taud;
}

void STPConnection::set_ujump(AurynFloat r) {
	Ujump = r;
}

void STPConnection::clear() 
{
	for (NeuronID i = 0; i < src->get_rank_size() ; i++)
	{
		   auryn_vector_float_set (state_x, i, 1 ); // TODO
		   auryn_vector_float_set (state_u, i, Ujump );
	}
}

void STPConnection::clone_parameters(STPConnection * con) 
{
	tau_d = con->tau_d;
	tau_f = con->tau_f;
	Ujump = con->Ujump;
	clear();
}
