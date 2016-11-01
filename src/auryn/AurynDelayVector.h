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

#ifndef AURYNDELAYVECTOR_H_
#define AURYNDELAYVECTOR_H_

#include <ctime>
#include <assert.h>
#include <vector>

#include "auryn_definitions.h"
#include "AurynVector.h"


namespace auryn {


	/*! \brief AurynDelayVector is a AurynVectorFloat which keeps its own history in a ring buffer
	 *
	 * Note that AurynDelayVector does not tap into the Auryn clock and the method advance() has to be run to store a
	 * copy in the vector queue.
	 *
	 * Public member methods:
	 *
	 * advance(): Stores corrent vector state into a queue
	 * mem_get(i,d)  : Retrieves element i from the d delayed version
	 * mem_ptr(d)    : Returns pointer to the d delayed version of the vector.
	 *
	 */
	class AurynDelayVector : public AurynVectorFloat 
	{
		private:
			typedef AurynVectorFloat super;

			int delay_buffer_size_;
			int memory_pos_;
			std::vector< AurynVectorFloat* > memory;

		public:
			/*! \brief Default constructor */
			AurynDelayVector(NeuronID n, unsigned int delay_buffer_size);

			/*! \brief Default destructor */
			virtual ~AurynDelayVector();

			/*! \brief Advances memory buffer by one step */
			void advance();

			/*! \brief Returns delayed state vector
			 *
			 * \param delay The delay in timesteps to retrieve. Value needs to be > 0, values smaller than zero will be interpreted as the max delay.
			 **/
			AurynVectorFloat * mem_get_vector(int delay=-1);

			/*! \brief Returns delayed element
			 *
			 * \param i The element to get
			 * \param delay The delay in timesteps to retrieve. Value needs to be > 0, values smaller than zero will be interpreted as the max delay.
			 **/
			AurynFloat mem_get(NeuronID i, int delay=-1);

			/*! \brief Returns pointer to delayed array element 
			 *
			 * \param delay The delay in timesteps to retrieve. Value needs to be > 0, values smaller than zero will be interpreted as the max delay.
			 */
			AurynVectorFloat * mem_ptr(int delay=-1);

			/*! \brief Resizes the vector and the buffer vectors */
			void resize(NeuronID new_size);

			/*! \brief Returns delay buffer size in units of AurynTime */
			AurynTime get_delay_size() { return delay_buffer_size_; }

	};

}


#endif /*AURYNDELAYVECTOR_H_*/
