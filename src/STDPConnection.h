/* 
* Copyright 2014 Friedemann Zenke
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
#include "DuplexConnection.h"
#include "EulerTrace.h"
#include "LinearTrace.h"
#include "SpikeDelay.h"

#define TRACE EulerTrace

using namespace std;


/*! \brief Double STDP All-to-All Connection
 *
 * This class implements a STDP connection with two time constants.
 */
class STDPConnection : public DuplexConnection
{

private:
	void init(AurynFloat eta, AurynFloat maxweight);
	void init_shortcuts();

protected:

	AurynFloat tau_pre1;
	AurynFloat tau_pre2;
	AurynFloat tau_post1;
	AurynFloat tau_post2;

	NeuronID * fwd_ind; 
	AurynWeight * fwd_data;

	NeuronID * bkw_ind; 
	AurynWeight ** bkw_data;

	AurynDouble hom_fudge;

	TRACE * tr_pre1;
	TRACE * tr_pre2;
	TRACE * tr_post1;
	TRACE * tr_post2,;

	void propagate_forward();
	void propagate_backward();
	void sort_spikes();
	AurynWeight dw_pre(NeuronID post);
	AurynWeight dw_post(NeuronID pre, NeuronID post);

public:
	AurynFloat Apre;
	AurynFloat Bpre;

	AurynFloat Apost;
	AurynFloat Bpost;

	AurynFloat w_min;
	AurynFloat w_max;

	bool stdp_active;

	STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			TransmitterType transmitter=GLUT);

	STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat eta=1, 
			AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT);

	STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05, 
			AurynFloat eta=1, 
			AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT,
			string name = "STDPConnection" );

	virtual ~STDPConnection();
	virtual void finalize();
	void free();

	void set_min_weight(AurynWeight min);
	void set_max_weight(AurynWeight max);

	AurynWeight get_wmin();

	virtual void propagate();
	virtual void evolve();

};

#endif /*STDPCONNECTION_H_*/
