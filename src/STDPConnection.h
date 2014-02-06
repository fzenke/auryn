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


using namespace std;


/*! \brief Doublet STDP All-to-All as implemented in NEST as stdp_synapse_hom 
 *
 * This class implements a range of doublet STDP rules.
 *
 */
class STDPConnection : public DuplexConnection
{

private:
	AurynFloat learning_rate;

	AurynFloat param_lambda;
	AurynFloat param_alpha;

	AurynFloat param_mu_plus;
	AurynFloat param_mu_minus;

	void init(AurynFloat lambda, AurynFloat maxweight);
	void init_shortcuts();

protected:

	AurynFloat tau_plus;
	AurynFloat tau_minus;

	NeuronID * fwd_ind; 
	AurynWeight * fwd_data;

	NeuronID * bkw_ind; 
	AurynWeight ** bkw_data;

	PRE_TRACE_MODEL * tr_pre;
	DEFAULT_TRACE_MODEL * tr_post;


	AurynFloat fudge_pot;
	AurynFloat fudge_dep;


	void propagate_forward();
	void propagate_backward();

	void compute_fudge_factors();

public:

	bool stdp_active;

	STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			TransmitterType transmitter=GLUT);

	STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat lambda=1e-5, 
			AurynFloat maxweight=0.1 , 
			TransmitterType transmitter=GLUT);

	STDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05, 
			AurynFloat lambda=0.01, 
			AurynFloat maxweight=100. , 
			TransmitterType transmitter=GLUT,
			string name = "STDPConnection" );

	void set_alpha(AurynFloat a);
	void set_lambda(AurynFloat l);

	void set_mu_plus(AurynFloat m);
	void set_mu_minus(AurynFloat m);

	void set_max_weight(AurynWeight w);

	virtual ~STDPConnection();
	virtual void finalize();
	void free();

	virtual void propagate();
	virtual void evolve();

};

#endif /*STDPCONNECTION_H_*/
