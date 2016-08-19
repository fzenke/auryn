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

#ifndef STDPWDCONNECTION_H_
#define STDPWDCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "DuplexConnection.h"
#include "EulerTrace.h"
#include "LinearTrace.h"
#include "SpikeDelay.h"


namespace auryn {


/*! \brief Doublet STDP All-to-All as implemented in NEST as stdp_synapse_hom 
 *
 * This class implements a range of doublet STDP rules including weight dependent 
 * ones (hence the wd suffix in he classn ame). 
 * It is meant to be similar to stdp_synapse_hom in NEST. 
 *
 */
class STDPwdConnection : public DuplexConnection
{

private:
	AurynWeight learning_rate;

	AurynWeight param_lambda;
	AurynWeight param_alpha;

	AurynWeight param_mu_plus;
	AurynWeight param_mu_minus;

	void init(AurynWeight lambda, AurynWeight maxweight);
	void init_shortcuts();

protected:

	AurynWeight tau_plus;
	AurynWeight tau_minus;

	NeuronID * fwd_ind; 
	AurynWeight * fwd_data;

	NeuronID * bkw_ind; 
	AurynWeight ** bkw_data;

	Trace * tr_pre;
	Trace * tr_post;


	AurynWeight fudge_pot;
	AurynWeight fudge_dep;


	void propagate_forward();
	void propagate_backward();

	void compute_fudge_factors();

public:

	bool stdp_active;

	STDPwdConnection(SpikingGroup * source, NeuronGroup * destination, 
			TransmitterType transmitter=GLUT);

	STDPwdConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynWeight lambda=1e-5, 
			AurynWeight maxweight=0.1 , 
			TransmitterType transmitter=GLUT);

	STDPwdConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynWeight sparseness=0.05, 
			AurynWeight lambda=0.01, 
			AurynWeight maxweight=100. , 
			TransmitterType transmitter=GLUT,
			string name = "STDPwdConnection" );

	void set_alpha(AurynWeight a);
	void set_lambda(AurynWeight l);

	void set_mu_plus(AurynWeight m);
	void set_mu_minus(AurynWeight m);

	void set_max_weight(AurynWeight w);

	virtual ~STDPwdConnection();
	virtual void finalize();
	void free();

	virtual void propagate();
	virtual void evolve();

};

}

#endif /*STDPWDCONNECTION_H_*/
