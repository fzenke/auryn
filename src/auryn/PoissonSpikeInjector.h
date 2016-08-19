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

#ifndef POISSONSPIKEINJECTOR_H_
#define POISSONSPIKEINJECTOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "PoissonGroup.h"

namespace auryn {

/*! \brief A PoissonGroup which can directly add its output spike to another SpikingGroup by piggy backing onto it
 *
 *
 * \todo Have to make sure this group has the same rank_lock structure as the piggy back group 
 */
class PoissonSpikeInjector : public PoissonGroup
{
private:
	typedef PoissonGroup super;

	SpikingGroup * target;


protected:
	
public:
	/*! \brief Standard constructor. 
	 * \param target_group is the group to whose output we add the random poisson spikes
	 * \param rate is the mean firing rate of the poisson neurons in the spike injector 
	 */
	PoissonSpikeInjector(SpikingGroup * target_group, AurynDouble rate=5. );

	/*! \brief Default destructor */
	virtual ~PoissonSpikeInjector();

	/*! \brief Standard evolve function */
	virtual void evolve();
};

}

#endif /*POISSONSPIKEINJECTOR_H_*/
