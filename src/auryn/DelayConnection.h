/* 
* Copyright 2014-2019 Friedemann Zenke
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

#ifndef DELAYCONNECTION_H_
#define DELAYCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "SparseConnection.h"
#include "SpikeDelay.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include <boost/archive/text_oarchive.hpp> 
#include <boost/archive/text_iarchive.hpp> 
#include <boost/archive/binary_oarchive.hpp> 
#include <boost/archive/binary_iarchive.hpp> 

namespace auryn {

/*! \brief DelayConnection implements a SparseConnection with adjustable delays.
 *
 * DelayConnection adds a delay to spikes from the src group. The delays of
 * DelayConnection are added to the delays already present in SpikingGroup.
 * Moreover, the delays are connection specific and can be interpreted as
 * dendritic delays.
 * The minimum possible delay is one timestep.
 *
 * \todo This is a prototype connection which needs testing.
 * */

class DelayConnection : public SparseConnection
{
private:
	SpikeDelay * src_dly;

	void init();
	void free();

protected:
	void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
	{
		SparseConnection::virtual_serialize(ar,version);
		ar & *src_dly;
	}

	void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
	{
		SparseConnection::virtual_serialize(ar,version);
		ar & *src_dly;
	}

public:
	/*! \brief The bare constructor for manual filling and constructing DelayConnection objects. 
	 *
	 * You should only use this if you know what you are doing. */
	DelayConnection(NeuronID rows, NeuronID cols);

	/*! \brief The default constructor for DelayConnection */
	DelayConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			AurynWeight weight, 
			AurynFloat sparseness=0.05, 
			TransmitterType transmitter=GLUT, 
			string name="DelayConnection");

	/*! \brief The default destructor */
	virtual ~DelayConnection();


	/*! \brief The required virtual propagate function for propagating spikes */
	virtual void propagate();

	/*! \brief Sets the delay in in units of auryn_timestep which is added 
	 * to all spikes from the src group. */
	void set_delay_steps(unsigned int delay); //!< Set delay in time steps

	/*! \brief Sets the delay in in units of seconds which 
	 * is added to all spikes from the src group. */
	void set_delay(double delay=1e-3); //!< Set delay in seconds
};

}

#endif /*DELAYCONNECTION_H_*/
