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

#ifndef STPCONNECTION_H_
#define STPCONNECTION_H_

#include "auryn_definitions.h"
#include "SparseConnection.h"
#include "SimpleMatrix.cpp"
#include <gsl/gsl_blas.h>

using namespace std;

class STPConnection : public SparseConnection
{
private:
	// STP parameters (maybe this should all move to a container)
	gsl_vector_float * state_x;
	gsl_vector_float * state_u;
	gsl_vector_float * state_temp;

	double tau_d;
	double tau_f;
	double Urest;
	double Ujump;


public:

	STPConnection(const char * filename);
	STPConnection(NeuronID rows, NeuronID cols);
	STPConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter=GLUT);
	STPConnection(SpikingGroup * source, NeuronGroup * destination, const char * filename , TransmitterType transmitter=GLUT);
	STPConnection(SpikingGroup * source, NeuronGroup * destination, AurynWeight weight, AurynFloat sparseness=0.05, TransmitterType transmitter=GLUT, string name="STPConnection");

	void set_tau_d(AurynFloat taud);
	void set_tau_f(AurynFloat tauf);
	void set_ujump(AurynFloat r);
	void set_urest(AurynFloat r);

	virtual ~STPConnection();
	virtual void propagate();
	void push_attributes();
	void evolve();
	void init();
	void free();

};

#endif /*STPCONNECTION_H_*/
