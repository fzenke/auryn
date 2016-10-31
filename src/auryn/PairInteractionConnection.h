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

#define PAIRINTERACTIONCON_WINDOW_MAX_SIZE 10000 //< Maximum STDP window size in auryn_timestep

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "DuplexConnection.h"

namespace auryn {


/*! \brief STDP Connection class to simulate arbitrary nearest-neighbor STDP windows.
 *
 * This class implements event-based STDP as nearest-neighbor spike interactions. 
 * Arbitrary STDP windows can be loaded from an external file or can be set by directly acessing the 
 * arrays window_pre_post and window_post_pre.
 *
 * \todo Add usage example and unit tests
 */
class PairInteractionConnection : public DuplexConnection
{

private:

	void init(AurynWeight maxw);
	void free();

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

	/*! \brief Switches stdp on or off */
	bool stdp_active;

	PairInteractionConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynWeight maxweight=1. , TransmitterType transmitter=GLUT);

	/*! \brief Default random sparse constructor */
	PairInteractionConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05,
			AurynWeight maxweight=1. , TransmitterType transmitter=GLUT, string name="PairInteractionConnection");

	/*! \brief Default destructor */
	virtual ~PairInteractionConnection();


	/*! \brief Loads STDP windows for pre-post and post-pre from ASCII file 
	 *
	 * \param filename String pointing to the file containing the STDP window
	 * \param scale Scales read in values by this number
	 * \todo Add file format description
	 */
	void load_window_from_file( const char * filename , double scale = 1. );

	/*! \brief Sets STDP window to be bi-exponential
	 *
	 * \param Aplus pre-post amplitude of window
	 * \param tau_plus pre-post time constant of window
	 * \param Aminus post-pre amplitude of window
	 * \param tau_minus post-pre time constant of window
	 */
	void set_exponential_window ( double Aplus = 1e-3, double tau_plus = 20e-3, double Aminus = -1e-3, double tau_minus = 20e-3);

	/*! \brief Sets "floor" terms for STDP rule
	 *
	 * These terms are used as update size when a time difference falls out of the window size.
	 *
	 * \param pre_post Floor for pre post pairs
	 * \param post_pre Floor for post pre pairs
	 */
	void set_floor_terms( double pre_post = 0.0, double post_pre = 0.0 );

	virtual void propagate();

};

}

#endif /*PAIRINTERACTIONCONNECTION_H_*/
