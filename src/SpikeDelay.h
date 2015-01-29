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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#ifndef SPIKEDELAY_H_
#define SPIKEDELAY_H_

#include "auryn_definitions.h"
// #include "SpikeContainer.h"
#include <vector>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

using namespace std;


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

		int ndelay;
		void free();

	public:


		SpikeDelay( int delaysteps = MINDELAY+1 );
		virtual ~SpikeDelay();

		void set_delay( int delay);
		void set_clock_ptr(AurynTime * clock);

		/*! Allows to insert spikes so many time steps ahead with less than max delay. */
		void insert_spike(NeuronID i, AurynTime ahead); 

		/*! Allows to use SpikeDelay like a queue. This pushes into get_spikes() */
		void push_back(NeuronID i); 

		/*! Returns the number of attributes per spike. */
		int get_num_attributes();

		/*! Used by SyncBuffer to submit x attributes with spikes in this delay. */
		void inc_num_attributes(int x);

		SpikeContainer * get_spikes(unsigned int pos = 1);
		SpikeContainer * get_spikes_immediate();

		AttributeContainer * get_attributes(unsigned int pos = 1);
		AttributeContainer * get_attributes_immediate();

		/*! Clears all containers in delay */
		void clear();
};


#endif /*SPIKEDELAY_H_*/
