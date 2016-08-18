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

#include "PoissonSpikeInjector.h"

using namespace auryn;

PoissonSpikeInjector::PoissonSpikeInjector(SpikingGroup * target_group, AurynDouble rate ) : PoissonGroup( target_group->get_size(), rate )
{
	target = target_group;
	if ( sys->mpi_size() > 1 ) {
		logger->warning("PoissonSpikeInjector might not work correctly when run with MPI.");
		// TODO still need to make sure that rank lock etc does not screw up things if the target is distributed differently
	}
}

PoissonSpikeInjector::~PoissonSpikeInjector()
{
}

void PoissonSpikeInjector::evolve()
{
	super::evolve();

	// now add Poisson spikes to target group
	SpikeContainer * a = target->get_spikes_immediate();
	SpikeContainer * b = get_spikes_immediate();
	a->insert(a->end(), b->begin(), b->end());
}
