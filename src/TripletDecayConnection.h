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

#ifndef TRIPLETDECAYCONNECTION_H_
#define TRIPLETDECAYCONNECTION_H_

#include "auryn_definitions.h"
#include "TripletConnection.h"
#include "EulerTrace.h"

#define TRIPLETDECAYCONNECTION_EULERUPGRADE_STEP 0.999

namespace auryn {


/*! \brief Implements triplet STDP with an exponential weight decay
 *
 * This is one of the Connection objects used for most simulations in Zenke et
 * al. 2013 to simulate large plastic recurrent networks with homeostatic
 * triplet STDP.
 */
class TripletDecayConnection : public TripletConnection
{

private:
	AurynTime decay_timestep;
	AurynFloat tau_decay;
	AurynFloat mul_decay;
	AurynWeight w_rest;
	AurynInt decay_count;

public:
	TripletDecayConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter);

	TripletDecayConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat tau_hom=10, 
			AurynFloat eta=1, AurynFloat decay = 1e-3,
			AurynFloat kappa=3., AurynWeight wrest=0., AurynWeight maxweight=1. , 
			TransmitterType transmitter=GLUT);

	TripletDecayConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05, 
			AurynFloat tau_hom=10, 
			AurynFloat eta=1, AurynFloat decay=1e-3,
			AurynFloat kappa=3., AurynWeight wrest=0., AurynWeight maxweight=1. , 
			TransmitterType transmitter=GLUT);

	virtual ~TripletDecayConnection();
	void init(AurynFloat decay, AurynWeight wrest);
	void free();

	virtual void propagate();

};

}

#endif /*TRIPLETDECAYCONNECTION_H_*/
