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
*/

#ifndef STDPCONNECTION_H_
#define STDPCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "DuplexConnection.h"
#include "Trace.h"
#include "LinearTrace.h"
#include "SpikeDelay.h"


namespace auryn {


/*! \brief Double STDP All-to-All Connection
 *
 * This class implements standard STDP with a double exponential window and optinal
 * offset terms. Window amplitudes and time constants are freely configurable.
 */
class STDPConnection : public DuplexConnection
{

private:
	void init(AurynFloat eta, AurynFloat tau_pre, AurynFloat tau_post, AurynFloat maxweight);

protected:

	AurynDouble hom_fudge;

	Trace * tr_pre;
	Trace * tr_post;

	void propagate_forward();
	void propagate_backward();

	AurynWeight on_pre(NeuronID post);
	AurynWeight on_post(NeuronID pre);

public:
	AurynFloat A; /*!< Amplitude of post-pre part of the STDP window */
	AurynFloat B; /*!< Amplitude of pre-post part of the STDP window */

	bool stdp_active;

	STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			TransmitterType transmitter=GLUT);

	STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat eta=1, 
			AurynFloat tau_pre=20e-3,
			AurynFloat tau_post=20e-3,
			AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT);

	STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05, 
			AurynFloat eta=1, 
			AurynFloat tau_pre=20e-3,
			AurynFloat tau_post=20e-3,
			AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT,
			string name = "STDPConnection" );


	virtual ~STDPConnection();
	virtual void finalize();
	void free();

	virtual void propagate();
	virtual void evolve();

};

}

#endif /*STDPCONNECTION_H_*/
