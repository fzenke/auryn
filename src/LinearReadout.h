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
*/

#ifndef LINEARREADOUT_H_
#define LINEARREADOUT_H_

#include "auryn_definitions.h"
#include "Monitor.h"
#include "System.h"
#include "Connection.h"
#include <fstream>
#include <iomanip>

using namespace std;

/*! \brief Records the membrane potential from one unit from the source neuron group to a file.*/
class LinearReadout : protected Monitor
{
protected:
	/*! The source neuron group to record from */
	NeuronGroup * src;
	/*! Number of perceptrons */
	int no_of_perceptrons;
	/*! The spike counts */
	gsl_vector_uint * count;
	/*! The weight vetors */
	vector<auryn_vector_float *> w;
	/*! The output vector */
	vector<double> y;
	/*! Falling edge of this guy triggers readout */
	bool * trigger;	
	/*! The step size (sampling interval) in units of dt */
	AurynTime ssize;
	/*! Standard initialization */
	void init(NeuronGroup * source, const char * filename, int np, AurynTime stepsize);
	/*! Computes the scalar products */
	void readout();
	/*! Standard destruction */
	void free();
	
public:
	LinearReadout(NeuronGroup * source, const char * filename, int np=10, AurynTime stepsize=100);
	virtual ~LinearReadout();
	void propagate();
};

#endif /*LINEARREADOUT_H_*/
