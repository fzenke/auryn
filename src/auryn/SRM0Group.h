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

#ifndef SRM0GROUP_H_
#define SRM0GROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "System.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

namespace auryn {

/*! \brief Implements SRM0 neuron model with escape noise
 *
 *
 * For a detailed introduction to the SRM and SRM0 neuron models see  Gerstner,
 * W., Kistler, W.M., Naud, R., and Paninski, L. (2014). Neuronal dynamics:
 * from single neurons to networks and models of cognition (Cambridge:
 * Cambridge University Press).
 *
 */
class SRM0Group : public NeuronGroup
{
private:
	static boost::mt19937 gen; 
	boost::exponential_distribution<> * dist;
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > * die;
	unsigned int salt;

	AurynFloat e_rest,e_rev,thr,tau_mem, tau_syn;
	AurynFloat scale_mem;
	AurynFloat scale_syn;

	/*! \brief Vector holding neuronspecific synaptic currents */
	AurynStateVector * syn_current;

	/*! \brief Warped TTL vector */
	AurynStateVector * warped_lifetime;

	/*! \brief Temporary vector */
	AurynStateVector * temp;

	void init();
	void calculate_scale_constants();
	inline void integrate_state();
	inline void check_thresholds();
public:
	/*! \brief Mean firing rate rate at threshold */
	AurynFloat rho0;

	/*! \brief Spike sharpness parameter delta u */
	AurynFloat delta_u;

	/*! The default constructor of this NeuronGroup */
	SRM0Group(NeuronID size);
	virtual ~SRM0Group();

	/*! \brief Redraws random waiting times neuron i */
	void draw(NeuronID i);

	/*! \brief Redraws random waiting times for all neurons */
	void draw_all();

	/*! Sets the membrane time constant (default 20ms) */
	void set_tau_mem(AurynFloat taum);

	/*! Resets all neurons to defined and identical initial state. */
	void clear();

	/*! The evolve method internally used by System. */
	void evolve();

	/*! \brief Seed the random number generator of all SRM0Group instances */
	void seed(unsigned int s);
};

}

#endif /*SRM0GROUP_H_*/

