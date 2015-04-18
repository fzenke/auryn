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

#ifndef SYMMETRICSTDPCONNECTION_H_
#define SYMMETRICSTDPCONNECTION_H_

#include "auryn_definitions.h"
#include "DuplexConnection.h"
#include "EulerTrace.h"
#include "LinearTrace.h"

using namespace std;


/*! \brief Implements a symmetric STDP window with an optional 
 *         presynaptic offset as used for inhibitory plasticity in Vogels et al. 2011.
 *
 * This class implements a plastic connection object implementing the plasticity rule
 * we used in Vogels et al. 2011 for the inhibitory plasticity.
 */
class modulatedSymmetricSTDPConnection : public DuplexConnection
{
private:
	ifstream modulation_file;
	AurynTime filetime;
	AurynFloat mods;
	AurynFloat newmods;

public:
	AurynFloat learning_rate;
	AurynFloat target;
	AurynFloat kappa_fudge;
    AurynFloat gmod;
	AurynWeight w_max;

	PRE_TRACE_MODEL * tr_pre;
	DEFAULT_TRACE_MODEL * tr_post;

	inline AurynWeight dw_pre(NeuronID post);
	inline AurynWeight dw_post(NeuronID pre);

	inline void propagate_forward();
	inline void propagate_backward();

	bool stdp_active;

	/*! Constructor to create a random sparse connection object and set up a temporally modulated symmteric plasticity rule, implementing event driven CD (Neftci et al 2014, Frontiers in Neuromorphic Engineering
	 *
	 * @param source the source group from where spikes are coming.
	 * @param destination the destination group where spikes are going.
	 * @param modulation_filename the filename modulating the STDP rule
	 * @param weight the initial weight of all connections.
	 * @param sparseness the connection probability for the sparse random set-up of the connections.
	 * @param eta the learning rate parameter.
	 * @param kappa the target rate paramter (alpha in the original publication).
	 * @param tau_stdp the size of one side of the STDP window.
	 * @param maxweight the maxium allowed weight.
	 * @param transmitter the transmitter type of the connection - by default GABA for inhibitory connection.
	 * @param name a meaningful name of the connection which will appear in debug output.
	 */
	modulatedSymmetricSTDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05,
			AurynFloat eta=1e-3, AurynFloat kappa=5., AurynFloat tau_stdp=5e-3,
			AurynWeight maxweight=10. , TransmitterType transmitter=GABA, string name="modulatedSymmetricSTDPConnection");

	modulatedSymmetricSTDPConnection(SpikingGroup * source, NeuronGroup * destination, const char * modulation_filename,
			AurynWeight weight, AurynFloat sparseness=0.05,
			AurynFloat eta=1e-3, AurynFloat kappa=5., AurynFloat tau_stdp=5e-3,
			AurynWeight maxweight=10. , TransmitterType transmitter=GABA, string name="modulatedSymmetricSTDPConnection");

	/*! Constructor that creates the connection directly from a wmat file.
	 *
	 * @param source the source group from where spikes are coming.
	 * @param destination the destination group where spikes are going.
	 * @param modulation_filename the filename modulating the STDP rule
	 * @param filename the filename of a wmat file to build he connection from
	 * @param eta the learning rate parameter.
	 * @param kappa the target rate paramter (alpha in the original publication).
	 * @param tau_stdp the size of one side of the STDP window.
	 * @param maxweight the maxium allowed weight.
	 * @param transmitter the transmitter type of the connection - by default GABA for inhibitory connection.
	 * @param name a meaningful name of the connection which will appear in debug output.
	 */
	modulatedSymmetricSTDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat eta=1e-3, AurynFloat kappa=5., AurynFloat tau_stdp=20e-3, 
			AurynWeight maxweight=10 , TransmitterType transmitter=GABA);

	modulatedSymmetricSTDPConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			const char * modulation_filename, 
			AurynFloat eta=1e-3, AurynFloat kappa=5., AurynFloat tau_stdp=20e-3, 
			AurynWeight maxweight=10 , TransmitterType transmitter=GABA);

	virtual ~modulatedSymmetricSTDPConnection();
	void init_mod(const char * modulation_filename, AurynFloat eta, AurynFloat kappa, AurynFloat tau_stdp, AurynWeight maxweight);
	void init(AurynFloat eta, AurynFloat kappa, AurynFloat tau_stdp, AurynWeight maxweight);
	void free();

	virtual void propagate();


};

#endif /*SYMMETRICSTDPCONNECTION_H_*/
