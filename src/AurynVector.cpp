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

#include "AurynVector.h"

using namespace auryn;

AurynVectorFloat::AurynVectorFloat(NeuronID n) : AurynVector<float>(n)
{

}

void AurynVectorFloat::scale(AurynFloat a) 
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	data[0:size:1] = a * data[0:size:1];
	#else
	const __m128 scalar = _mm_set1_ps(a);
	for ( float * i = data ; i != data+size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( i );
		__m128 result = _mm_mul_ps(chunk, scalar);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < b->size ; ++i ) {
		data[i] *= a;
	}
#endif /* CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY */
}


void AurynVectorFloat::saxpy(AurynFloat a, AurynVector<float> * x)
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	data[0:size:1] = a * x->data[0:x->size:1] + data[0:size:1];
	#else
	float * xp = x->data;
	const __m128 alpha = _mm_set1_ps(a);
	for ( float * i = data ; i < data+size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( xp ); xp += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result     = _mm_mul_ps( alpha, chunk );

		chunk  = sse_load( i );
		result = _mm_add_ps( result, chunk );
		sse_store( i, result ); 
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < size ; ++i ) {
		data[i] += a * x->data[i];
	}
#endif
}


void AurynVectorFloat::clip(float min, float max) 
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	for ( NeuronID i = 0 ; i < size ; ++i ) {
		if ( data[i] < min ) {
			data[i] = min;
		} else 
			if ( data[i] > max ) 
				data[i] = max;
	}
	#else
	const __m128 lo = _mm_set1_ps(min);
	const __m128 hi = _mm_set1_ps(max);
	for ( float * i = data ; i != data+size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( i );
		__m128 result = _mm_min_ps(chunk, hi);
		result = _mm_max_ps(result, lo);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < size ; ++i ) {
		if ( data[i] < min ) {
			data[i] = min;
		} else 
			if ( data[i] > max ) 
				data[i] = max;
	}
#endif
}

void AurynVectorFloat::mul(AurynVector<float> * v) 
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	data[0:size:1] = data[0:size:1] * v->data[0:v->size:1];
	#else
	float * bd = v->data;
	for ( float * i = data ; i != data+size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk_a = sse_load( i );
		__m128 chunk_b = sse_load( bd ); bd+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_mul_ps(chunk_a, chunk_b);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < size ; ++i ) {
		data[i] *= v->data[i];
	}
#endif
}
