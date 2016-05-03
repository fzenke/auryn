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

#ifndef TRIPLETSCALINGCONNECTION_H_
#define TRIPLETSCALINGCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "DuplexConnection.h"
#include "EulerTrace.h"

#define TRIPLETSCALINGCONNECTION_EULERUPGRADE_STEP 0.001


namespace auryn {
	class TripletScalingConnection : public DuplexConnection
	{

	private:
		void init(AurynFloat tau_hom, AurynFloat eta, AurynFloat kappa, AurynFloat beta, AurynFloat maxweight);
		void init_shortcuts();

		virtual AurynWeight get_hom(NeuronID i);

	protected:

		AurynFloat tau_plus;
		AurynFloat tau_minus;
		AurynFloat tau_long;

		AurynFloat tau_homeostatic;

		AurynTime scal_timestep;
		AurynFloat scal_beta;
		AurynFloat scal_mul;

		NeuronID * fwd_ind; 
		AurynWeight * fwd_data;

		NeuronID * bkw_ind; 
		AurynWeight ** bkw_data;

		AurynDouble hom_fudge;
		AurynDouble target_rate;

		PRE_TRACE_MODEL * tr_pre;
		DEFAULT_TRACE_MODEL * tr_post;
		DEFAULT_TRACE_MODEL * tr_post2;
		DEFAULT_TRACE_MODEL * tr_post_hom;

		void propagate_forward();
		void propagate_backward();
		inline void evolve_scaling();
		void sort_spikes();

		AurynWeight dw_pre(NeuronID post);
		AurynWeight dw_post(NeuronID pre, NeuronID post);

	public:
		AurynFloat A3_plus;

		AurynFloat w_min;
		AurynFloat w_max;



		bool stdp_active;

		TripletScalingConnection(SpikingGroup * source, NeuronGroup * destination, 
				TransmitterType transmitter=GLUT);

		TripletScalingConnection(SpikingGroup * source, NeuronGroup * destination, 
				const char * filename, 
				AurynFloat tau_hom=10, 
				AurynFloat eta=1, 
				AurynFloat kappa=3., 
				AurynFloat beta=1.0, // TODO put sensical number
				AurynFloat maxweight=1. , 
				TransmitterType transmitter=GLUT);

		TripletScalingConnection(SpikingGroup * source, NeuronGroup * destination, 
				AurynWeight weight, AurynFloat sparseness=0.05, 
				AurynFloat tau_hom=10, 
				AurynFloat eta=1, 
				AurynFloat kappa=3., 
				AurynFloat beta=1.0, // TODO put sensical number
				AurynFloat maxweight=1. , 
				TransmitterType transmitter=GLUT,
				string name = "TripletScalingConnection" );

		virtual ~TripletScalingConnection();
		virtual void finalize();
		void free();

		void set_min_weight(AurynWeight min);
		void set_max_weight(AurynWeight max);
		void set_hom_trace(AurynFloat freq);
		void set_beta(AurynFloat beta);

		AurynWeight get_wmin();

		virtual void propagate();
		virtual void evolve();

	};
}

#endif /*TRIPLETSCALINGCONNECTION_H_*/
