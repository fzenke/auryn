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

#ifndef PAIRINTERACTIONCONNECTION_H_
#define PAIRINTERACTIONCONNECTION_H_

#define WINDOW_MAX_SIZE 60000

#include "auryn_definitions.h"
#include "DuplexConnection.h"
#include "EulerTrace.h"

namespace auryn {


/*! \brief STDP Connection class to simulate arbitrary nearest-neighbor STDP windows.
 */
class PairInteractionConnection : public DuplexConnection
{

protected:
	AurynWeight w_max;

	AurynTime * last_spike_pre;
	AurynTime * last_spike_post;

	inline AurynWeight dw_fwd(NeuronID post);
	inline AurynWeight dw_bkw(NeuronID pre);

	inline void propagate_forward();
	inline void propagate_backward();


public:
	AurynFloat * window_pre_post;
	AurynFloat * window_post_pre;

	bool stdp_active;

	PairInteractionConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynWeight maxweight=1. , TransmitterType transmitter=GLUT);

	PairInteractionConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05,
			AurynWeight maxweight=1. , TransmitterType transmitter=GLUT, string name="PairInteractionConnection");
	virtual ~PairInteractionConnection();
	void init(AurynWeight maxw);
	void free();

	void load_window_from_file( const char * filename , double scale = 1. );
	void set_exponential_window ( double Aplus = 1e-3, double tau_plus = 20e-3, double Aminus = -1e-3, double tau_minus = 20e-3);
	void set_floor_terms( double pre_post = 0.0, double post_pre = 0.0 );

	virtual void propagate();

};

}

#endif /*PAIRINTERACTIONCONNECTION_H_*/
