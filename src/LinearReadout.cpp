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

#include "LinearReadout.h"

LinearReadout::LinearReadout(NeuronGroup * source, const char * filename, int np, AurynTime stepsize) : Monitor(filename)
{
	init(source,filename,stepsize);
}


void LinearReadout::init(NeuronGroup * source, const char * filename, int np, AurynTime stepsize)
{
	sys->register_monitor(this);
	src = source;
	ssize = stepsize;
	no_of_perceptrons = np;
	outfile << setiosflags(ios::fixed) << setprecision(6);

	count = gsl_vector_uint_alloc( source->get_rank_size()+1 );
	gsl_vector_uint_set_basis( count, no_of_perceptrons );
	for ( int i = 0 ; i < no_of_perceptrons ; ++i ) {
		w.push_back( gsl_vector_float_alloc( source->get_rank_size()+1 ) );
		// gsl_vector_float_set_zero( w[i] );
	}
}

LinearReadout::free()
{
	gsl_vector_uint_free( count );
	for ( int i = 0 ; i < no_of_perceptrons ; ++i )
		gsl_vector_float_free(w[i]);
}

LinearReadout::~LinearReadout()
{
	free();
}

void LinearReadout::readout()
{
	for ( int i = 0 ; i < no_of_perceptrons ; ++i ) {
		gsl_blas_dsdot( count , w[i] , y[i] );
	}
}

void LinearReadout::propagate()
{
	// count spikes
	for (NeuronID * spike = source->get_spikes_immediate()->begin() ; 
			spike != source->get_spikes_immediate()->end() ; 
			++spike ) {
		count->data+*spike += 1;
	}
	if ((sys->get_clock())%ssize==0) { // TODO
		outfile << dt*(sys->get_clock()) << " " ;
		readout();
		for ( int i = 0 ; i < no_of_perceptrons ; ++i ) {
			outfile << y[i] << " ";
		}
		outfile << "\n";
		gsl_vector_uint_set_basis( count, no_of_perceptrons );
	}
}
