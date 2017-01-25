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

#ifndef SPIKEDELAY_H_
#define SPIKEDELAY_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include <vector>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

namespace auryn {


/*! \brief Delay object for spikes which is synchronized between
 *         nodes using the SyncBuffer formalism implemented in System.
 *
 * A simple list class to store spikes (numbers of type NeuronID). 
 * Memory allocation is only done when the container needs to grow. The class never shrinks 
 * thus optimizing performance but not memory efficiency. 
 */

class SpikeDelay
{
	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			for (unsigned int i = 0 ; i < ndelay ; ++i ) {
				SpikeContainer * sc = get_spikes(i);
				ar & *sc;
			}

			for (unsigned int i = 0 ; i < ndelay ; ++i ) {
				AttributeContainer * ac = get_attributes(i);
				ar & *ac;
			}
		}

		SpikeContainer ** delaybuf;
		AttributeContainer ** attribbuf;

		int numSpikeAttributes;

		static AurynTime * clock_ptr;

		unsigned int ndelay;
		void free();

	public:


		/*! \brief The default constructor. */
		SpikeDelay( int delaysteps = MINDELAY+1 );

		/*! \brief The default destructor. */
		virtual ~SpikeDelay();

		/*! \brief Set delay in number of timesteps. 
		 *
		 * This allows to set the size of the delay in timesteps. The delay has to be at least of
		 * size MINDELAY+1. */
		void set_delay( int delay );

		/*! \brief Get delay time in AurynTime
		 *
		 */
		int get_delay( );

		/*! \brief Internal function to set clock pointer
		 *
		 * Sets internal clock pointer to system wide clock pointer. It is used 
		 * by the System class */
		void set_clock_ptr(AurynTime * clock);

		/*! \brief Allows to insert spikes so many time steps ahead with less than max delay. */
		void insert_spike(NeuronID i, AurynTime ahead); 

		/*! \brief Allows to use SpikeDelay like a queue. 
		 *
		 * This pushes into get_spikes_immediate() */
		void push_back(NeuronID i); 

		/*! \brief Pushes all elemens from given SpikeContainer into the delay
		 *
		 * This pushes into get_spikes_immediate() */
		void push_back( SpikeContainer * sc ); 

		/*! \brief Returns the number of spike attributes per spike. 
		 *
		 * Spike attributes are used to implement short-term plasticity or similar mechanisms efficiently.*/
		int get_num_attributes();

		/*! \brief Internally used by SyncBuffer to submit x attributes with spikes in this delay. */
		void inc_num_attributes(int x);

		/*! \brief Returns the spikes at a given delay position.
		 *
		 * pos == 1 corresponds to the maximum delay of the SpikeDelay and at 
		 * least to MINDELAY+1. pos == 2 corresponds to the maximum delay -1,
		 * and so forth ...*/
		SpikeContainer * get_spikes(unsigned int pos = 1);

		/*! \brief Returns the spikes stored into the delay within this very same time step. */
		SpikeContainer * get_spikes_immediate();

		/*! \brief Like get_spikes but returns the spike attributes. */
		AttributeContainer * get_attributes(unsigned int pos = 1);

		/*! \brief Like get_spikes_immediate but returns the spike attributes. */
		AttributeContainer * get_attributes_immediate();

		/*! \brief Print delay contents for debugging . */
		void print();

		/*! Clears all containers in delay. */
		void clear();
};

}


#endif /*SPIKEDELAY_H_*/
