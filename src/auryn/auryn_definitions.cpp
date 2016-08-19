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

#include "auryn_definitions.h"
#include "AurynVector.h" // required for forward declaration of template

namespace auryn {

double auryn_timestep = 1e-4;

int auryn_AlignOffset // copied from ATLAS
(const int N,       /* max return value */
const void *vp,    /* pointer to be aligned */
const int inc,     /* size of each elt, in bytes */
const int align)   /* required alignment, in bytes */
{
	const int p = align/inc;
	const size_t k=(size_t)vp, j=k/inc;
	int iret;
	if (k == (j)*inc && p*inc == align)
	{
		iret = ((j+p-1) / p)*p - j;
		if (iret <= N) return(iret);
	}
	return(N);
}

NeuronID calculate_vector_size(NeuronID i)
{
	if ( i%SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS==0 ) 
		return i;
	NeuronID div = i/SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS; // rounds down
	NeuronID new_size = (div+1)*SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS; // is multiple of SIMD...
	return new_size;
}


void auryn_vector_float_mul( auryn_vector_float * a, auryn_vector_float * b)
{
	a->mul(b);
}

void auryn_vector_float_add_constant( auryn_vector_float * a, const float b )
{
	a->add(b);
}

void auryn_vector_float_scale( const float a, auryn_vector_float * b )
{
	b->scale(a);
}

void auryn_vector_float_saxpy( const float a, auryn_vector_float * x, auryn_vector_float * y )
{
	y->saxpy(a,x);
}

void auryn_vector_float_add( auryn_vector_float * a, auryn_vector_float * b)
{
	a->add(b);
}

void auryn_vector_float_sub( auryn_vector_float * a, auryn_vector_float * b)
{
	a->sub(b);
}

void auryn_vector_float_sub( auryn_vector_float * a, auryn_vector_float * b, auryn_vector_float * r)
{
	r->diff(a,b);
}

void auryn_vector_float_clip( auryn_vector_float * v, const float a, const float b ) {
	v->clip(a, b);
}

void auryn_vector_float_clip( auryn_vector_float * v, const float a ) {
	v->clip(a,0.0);
}

auryn_vector_float * auryn_vector_float_alloc( const NeuronID n ) {
	return new auryn_vector_float(n);
}

void auryn_vector_float_free ( auryn_vector_float * v ) {
	delete v;
}

void auryn_vector_float_set_all ( auryn_vector_float * v, AurynFloat x ) {
	v->set_all(x);
}

void auryn_vector_float_set_zero ( auryn_vector_float * v ) {
	v->set_zero();
}

AurynFloat auryn_vector_float_get ( auryn_vector_float * v, const NeuronID i ) {
	return v->data[i];
}

AurynFloat * auryn_vector_float_ptr ( auryn_vector_float * v, const NeuronID i ) {
	return v->data+i;
}

void auryn_vector_float_set ( auryn_vector_float * v, const NeuronID i, AurynFloat x ) {
	v->data[i] = x;
}

void auryn_vector_float_copy ( auryn_vector_float * src, auryn_vector_float * dst ) {
	dst->copy(src);
}


auryn_vector_ushort * auryn_vector_ushort_alloc( const NeuronID n ) {
	return new auryn_vector_ushort(n);
}

void auryn_vector_ushort_free ( auryn_vector_ushort * v ) {
	delete v;
}

void auryn_vector_ushort_set_all ( auryn_vector_ushort * v, unsigned short x ) {
	v->set_all(x);
}

void auryn_vector_ushort_set_zero ( auryn_vector_ushort * v ) {
	v->set_zero();
}

unsigned short auryn_vector_ushort_get ( auryn_vector_ushort * v, const NeuronID i ) {
	return v->data[i];
}

unsigned short * auryn_vector_ushort_ptr ( auryn_vector_ushort * v, const NeuronID i ) {
	return v->data+i;
}

void auryn_vector_ushort_set ( auryn_vector_ushort * v, const NeuronID i, unsigned short x ) {
	v->data[i] = x;
}

void auryn_vector_ushort_copy ( auryn_vector_ushort * src, auryn_vector_ushort * dst ) {
	for ( NeuronID i = 0 ; i < dst->size ; ++i ) 
		dst->data[i] = src->data[i];
}

}
