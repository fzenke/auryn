/* 
* Copyright 2014-2015 Friedemann Zenke
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

#ifndef CHECKER_H_
#define CHECKER_H_

#include "auryn_definitions.h"
#include "SpikingGroup.h"

using namespace std;

class System;

/*! \brief The abstract base class for all checkers
 *
 * Checkers are online monitors that can be used to break a run.
 * In many simulations involving plasticity it is for instance 
 * useful to monitor the average firing rate of the network. If 
 * it drops to zero or explodes this can trigger the abortion of
 * the run to avoid hammering of the fileserver or other loss
 * of resources. The most dominant checker for this purpose is
 * RateChecker that should be included in every simulation.
 */
class Checker
{
private:
	/* Functions necesssary for serialization */
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		virtual_serialize(ar, version);
	}

	virtual void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) = 0;
	virtual void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) = 0;

protected:
	SpikingGroup * src;
	
public:
	Checker(SpikingGroup * source);
	virtual ~Checker();
	/*! The propagate function of Checkers is for internal use. 
	 * It is called by System and returns true to signal a break. 
	 * If checking is enabled for it will stop the current run.
	 */
	virtual bool propagate() = 0 ;
	/*! The propagate function of Checkers returns a bool value.
	 * When true this signals a break to System, which if checking
	 * is enabled will stop the current run.
	 */
	virtual AurynFloat get_property() = 0 ;
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Checker)

#endif /*CHECKER_H_*/
