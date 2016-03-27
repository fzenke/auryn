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

#ifndef FILEMODULATEDGROUP_H_
#define FILEMODULATEDGROUP_H_

#include "auryn_definitions.h"
#include "System.h"
#include "SpikingGroup.h"
#include "PoissonGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

namespace auryn {

/*! \brief A special Poisson generator that reads its instantaneous
 * firing rate from a tiser file. Datapoints in the rate file are
 * interpolated linearly.
 */
class FileModulatedPoissonGroup : public PoissonGroup
{
private:
	AurynTime ftime;
	AurynTime ltime;
	AurynDouble rate_m;
	AurynDouble rate_n;
	AurynDouble last_rate;

	char buffer[255];
	bool stimulus_active;

	std::ifstream inputfile;

	void init ( string filename );
	
public:
	FileModulatedPoissonGroup(NeuronID n, string filename );
	virtual ~FileModulatedPoissonGroup();
	virtual void evolve();
};

} // namespace 

#endif /*FILEMODULATEDGROUP_H_*/
