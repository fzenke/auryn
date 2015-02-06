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

#ifndef RATEMODULATEDCONNECTION_H_
#define RATEMODULATEDCONNECTION_H_

#include "auryn_definitions.h"
#include "SparseConnection.h"
#include "SimpleMatrix.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>

using namespace std;

typedef SimpleMatrix<AurynWeight*> BackwardMatrix;

//! \brief Rate Modulated Connection implements a SparseConnection in which the weights depend
//  	   on the averaged rate of a given SpikingGropu (rate_modulating_group).
//  
//  Rate modulation is happening with a global scaling factor (rate_modulation_mul) that
//  is set to be rate_modulation_mul = pow(rate_estimate/rate_target,rate_modulation_exponent).
//  Negative exponants are possible. Default behavior is optimized for GABA-ergic connections
//  (i.e. positive exponent of 2 increases connection strength if the rate is higher).
//*/
class RateModulatedConnection : public SparseConnection
{
private:
	AurynDouble rate_estimate_tau;
	AurynDouble rate_estimate_decay_mul;

	AurynDouble rate_modulation_scale;
	SpikingGroup * rate_modulating_group;

	void init();
	void free();

	/*! Controls the strength and sign of the response in the integral controller */
	AurynDouble eta;

public:
	bool stdp_active;
	AurynDouble rate_estimate;
	/*! Defines the rate target at which the modulation factor is 1 */ 
	AurynDouble rate_target; 
	/*! Defines the modulation exponent. */ 
	AurynDouble rate_modulation_exponent; 

	/*! The modulation value. */ 
	AurynDouble rate_modulation_mul;

	RateModulatedConnection(const char * filename);
	RateModulatedConnection(NeuronID rows, NeuronID cols);
	RateModulatedConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			TransmitterType transmitter=GLUT);
	RateModulatedConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			const char * filename , 
			TransmitterType transmitter=GLUT);
	RateModulatedConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			AurynWeight weight, 
			AurynFloat sparseness=0.05, 
			TransmitterType transmitter=GLUT, 
			string name="RateModulatedConnection");
	virtual ~RateModulatedConnection();

	virtual void stats(AurynFloat &mean, AurynFloat &std);

	void set_eta(AurynFloat value);

	void propagate_forward();
	void propagate();
	void evolve();
	void set_modulating_group(SpikingGroup * group);

	virtual bool load_from_file(string filename);
	virtual bool write_to_file(string filename);
};

#endif /*RATEMODULATEDCONNECTION_H_*/
