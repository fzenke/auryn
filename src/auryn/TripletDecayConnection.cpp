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

#include "TripletDecayConnection.h"

using namespace auryn;

void TripletDecayConnection::init(AurynFloat decay, AurynWeight wrest)
{
	tau_decay = decay;
	w_rest = wrest;
	mul_decay = TRIPLETDECAYCONNECTION_EULERUPGRADE_STEP;

	decay_timestep = -log(TRIPLETDECAYCONNECTION_EULERUPGRADE_STEP)*tau_decay/auryn_timestep;
	decay_count = decay_timestep;

	std::stringstream oss;
	oss << "TripletDecayConnection: (" << get_name() << "):"
		<< " decay_timestep= " << decay_timestep 
		<< ", mul_decay= " << mul_decay;
	auryn::logger->msg(oss.str(),VERBOSE);
}

void TripletDecayConnection::free()
{
}


TripletDecayConnection::TripletDecayConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) : TripletConnection(source, destination, transmitter)
{
}

TripletDecayConnection::TripletDecayConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynFloat tau_hom, 
		AurynFloat eta, AurynFloat decay,
		AurynFloat kappa, AurynWeight wrest, AurynWeight maxweight , 
		TransmitterType transmitter) : TripletConnection( source, destination, 
			filename, 
			tau_hom, 
			eta, 
			kappa, maxweight , 
			transmitter) 

{
	init(decay/eta,wrest);
}

TripletDecayConnection::TripletDecayConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat tau_hom, 
		AurynFloat eta, AurynFloat decay,
		AurynFloat kappa, AurynWeight wrest, AurynWeight maxweight , 
		TransmitterType transmitter) : TripletConnection( source, destination, 
			weight, sparseness, 
			tau_hom, 
			eta, 
			kappa, maxweight , 
			transmitter )
{
	init(decay/eta,wrest);
}

TripletDecayConnection::~TripletDecayConnection()
{
	free();
}

void TripletDecayConnection::propagate()
{
	TripletConnection::propagate();
	// decay of weights
	if ( stdp_active ) {
		if ( decay_count == 0 ) {
			for ( AurynWeight * i = w->get_data_begin() ; i != w->get_data_end() ; ++i ) {
				// *i *= mul_decay;
				*i = w_rest + mul_decay*(*i-w_rest);
			}
			decay_count = decay_timestep;
		}
		else 
			decay_count--;
	}
}

