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

#ifndef CORRELATEDPOISSONGROUP_H_
#define CORRELATEDPOISSONGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "SpikingGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

namespace auryn {

/*! \brief A PoissonGroup with multiple subpopulations that co-modulate their firing
 *         rate according to an Ornstein Uhlenbeck process.
 *
 * The group can be used to provide a stimulus for Hebbian learning in
 * networks. When defined a given number of groups of given size are defined,
 * as well as a correlation time. 
 */
class CorrelatedPoissonGroup : public SpikingGroup
{
private:
	AurynTime * clk;
	static boost::mt19937 shared_noise_gen; 
	static boost::mt19937 rank_noise_gen; 

	boost::uniform_01<> * dist;
	boost::variate_generator<boost::mt19937&, boost::uniform_01<> > * shared_noise;
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > * rank_noise;

	void init(AurynDouble rate, NeuronID gsize, AurynDouble timedelay );

protected:
	AurynDouble lambda;

	AurynTime tstop;

	NeuronID groupsize;
	NeuronID remainersize;
	NeuronID ngroups;

	AurynDouble timescale;
	int offset;
	AurynTime delay;

	/*! Stores Ornstein Uhlenbeck state */
	AurynDouble o;
	AurynDouble * delay_o;

	/*! Stores parameters for moving amplitude for slow changes to correlations structure */
	AurynDouble amplitude;
	AurynDouble tau_amplitude;
	AurynDouble target_amplitude;
	AurynDouble thr;

	/*! Stores the hopping state */
	NeuronID * x;
	
public:
	AurynDouble mean;
	/*! Default constructor.
	 *
	 * @param n the size of the group which will be devided by the size of one subgroup to give the number of groups.
	 * @param rate the mean firing rate of all cells.
	 * @param gsize the size of one subgroup.
	 * @param timedelay delay between groups.
	 */
	CorrelatedPoissonGroup(NeuronID n, 
			AurynDouble rate=5., 
			NeuronID gsize=100, 
			AurynDouble timedelay=50e-3 );
	virtual ~CorrelatedPoissonGroup();
	virtual void evolve();
	void set_rate(AurynDouble rate);
	void set_amplitude(AurynDouble ampl);
	void set_target_amplitude(AurynDouble ampl);
	void set_tau_amplitude(AurynDouble scale);
	void set_timescale(AurynDouble scale);
	void set_offset(int off);
	void set_threshold(AurynDouble threshold);
	void set_stoptime(AurynDouble stoptime);
	AurynDouble get_rate();
	void seed();
};

}

#endif /*NEURONGROUP_H_*/
