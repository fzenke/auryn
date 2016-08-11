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

#ifndef STPCONNECTION_H_
#define STPCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "SparseConnection.h"


namespace auryn {



/*! \brief This class implements short term plasticity according to the Tsodyks-Markram synapse 
 * 
 * This class implements the short-term plasticity model following 
 * Markram, H., Wang, Y., Tsodyks, M., 1998. Differential signaling via the same axon of neocortical pyramidal neurons. Proc Natl Acad Sci U S A 95, 5323–5328.
 *
 * also see 
 * Mongillo, G., Barak, O., Tsodyks, M., 2008. Synaptic Theory of Working Memory. Science 319, 1543–1546. doi:10.1126/science.1150769
 *
 */

class STPConnection : public SparseConnection
{
private:
	// STP parameters (maybe this should all move to a container)
	auryn_vector_float * state_x;
	auryn_vector_float * state_u;
	auryn_vector_float * state_temp;

	double tau_d;
	double tau_f;
	double Urest;
	double Ujump;


	void init();
	void free();

public:

	/*! Minimal constructor for from file init -- deprecated
	 * \param filename Filename to load
	 */
	STPConnection(const char * filename);

	/*! Minimal constructor to which leaves the connection uninitialized 
	 * \param source The presynaptic SpikingGroup
	 * \param destination the postsynaptic NeuronGroup
	 */
	STPConnection(NeuronID rows, NeuronID cols);

	/*! Default constructor to which leaves the connection uninitialized 
	 * \param source The presynaptic SpikingGroup
	 * \param destination the postsynaptic NeuronGroup
	 * \param transmitter The transmitter type
	 */
	STPConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter=GLUT);

	/*! Default constructor to initialize connection from file
	 * \param source The presynaptic SpikingGroup
	 * \param destination the postsynaptic NeuronGroup
	 * \param filename The file to load the connectivity from upon initialization
	 * \param transmitter The transmitter type
	 * \param name The connection name as it appears in debugging output
	 */
	STPConnection(SpikingGroup * source, NeuronGroup * destination, const char * filename , TransmitterType transmitter=GLUT);

	/*! Default constructor to initialize connection with random sparse connectivity 
	 * \param source The presynaptic SpikingGroup
	 * \param destination the postsynaptic NeuronGroup
	 * \param weight The default weight for connections 
	 * \param sparseness The probability of a connection for the sparse connectivity
	 * \param transmitter The transmitter type
	 * \param name The connection name as it appears in debugging output
	 */
	STPConnection(SpikingGroup * source, NeuronGroup * destination, AurynWeight weight, AurynFloat sparseness=0.05, TransmitterType transmitter=GLUT, string name="STPConnection");

	/*! Setter for tau_d, the timescale of synaptic depression. */
	void set_tau_d(AurynFloat taud);

	/*! Setter for tau_f, the timescale of synaptic facilitation. */
	void set_tau_f(AurynFloat tauf);

	/*! Setter for U_jump, which specifies the spike triggered change in the U equation in the model. Typically the same as urest. */
	void set_ujump(AurynFloat r);

	/*! Setter for U_rest the resting value of U. */
	void set_urest(AurynFloat r);

	/*! Default destructor. */
	virtual ~STPConnection();

	/*! Implements the connections propagate function (auryn internal use). */
	virtual void propagate();

	/*! Internal function to push spike attributes. */
	void push_attributes();

	/*! Implements the connections evolve function (auryn internal use). */
	void evolve();

};

}

#endif /*STPCONNECTION_H_*/
