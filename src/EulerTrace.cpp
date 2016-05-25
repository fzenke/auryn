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

#include "EulerTrace.h"

using namespace auryn;

void EulerTrace::init(NeuronID n, AurynFloat timeconstant)
{
	size = n;
	set_timeconstant(timeconstant);
	state = new AurynVectorFloat ( calculate_vector_size(size) ); 
	temp = new AurynVectorFloat ( calculate_vector_size(size) ); // temp vector
	set_all(0.);
	target_ptr = NULL;
}

void EulerTrace::free()
{
	delete state;
	delete temp;
}

EulerTrace::EulerTrace(NeuronID n, AurynFloat timeconstant)
{
	init(n,timeconstant);
}

EulerTrace::~EulerTrace()
{
	free();
}

void EulerTrace::set_timeconstant(AurynFloat timeconstant)
{
	tau = timeconstant;
	scale_const = exp(-dt/tau);
}

void EulerTrace::set(NeuronID i , AurynFloat value)
{
	state->set( i, value);
}

void EulerTrace::set_all(AurynFloat value)
{
	state->set_all(value);
}

void EulerTrace::add(AurynVectorFloat * values)
{
   state->add( values );
}

void EulerTrace::add(NeuronID i, AurynFloat value)
{
   // auryn_vector_float_set (state, i, auryn_vector_float_get (state, i) + value);
   state->data[i] += value;
}

void EulerTrace::set_target( AurynVectorFloat * target )
{
	if ( target != NULL ) {
	}
	target_ptr = target ;
}

void EulerTrace::set_target( EulerTrace * target )
{
	set_target( target->state );
}

void EulerTrace::evolve()
{
    // auryn_vector_float_scale(scale_const,state); // seems to be faster
	state->scale(scale_const);
	// auryn_vector_float_mul_constant(state,scale_const);
}

void EulerTrace::follow()
{ 
	temp->copy( state );
	auryn_vector_float_saxpy( -1., target_ptr, temp );
	auryn_vector_float_saxpy( -dt/tau, temp, state );
}

AurynVectorFloat * EulerTrace::get_state_ptr()
{
	return state;
}

AurynFloat EulerTrace::get_tau()
{
	return tau;
}

void EulerTrace::inc(NeuronID i)
{
   // auryn_vector_float_set (state, i, auryn_vector_float_get (state, i)+1);
   state->data[i]++;
}


AurynFloat EulerTrace::normalized_get(NeuronID i)
{
	return state->get( i ) / tau ;
}


void EulerTrace::clip(AurynState value)
{
	state->clip( 0.0, value);
}
