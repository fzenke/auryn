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

// TODO on the long run we should get rid of the unaligned instructions
inline __m128 sse_load( float * i ) 
{
#ifdef CODE_ALIGNED_SIMD_INSTRUCTIONS
	return _mm_load_ps( i );
#else
	return _mm_loadu_ps( i );
#endif
}

inline void sse_store( float * i, __m128 d ) 
{
#ifdef CODE_ALIGNED_SIMD_INSTRUCTIONS
	_mm_store_ps( i, d );
#else
	_mm_storeu_ps( i, d );
#endif
}


AurynVectorFloat::AurynVectorFloat(NeuronID n) : AurynVector<float>(n)
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	// check that size is a multiple of SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS
	// which is typically 4 for float and SSE
	if ( n%SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS ) {
		resize(n);
	}
#endif /* CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY */
}

void AurynVectorFloat::resize(NeuronID new_size) 
{
	if ( new_size%SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS ) {
		const NeuronID div = new_size/SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS; // rounds down
		new_size = (div+1)*SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS; // is multiple of SIMD...
	}
	super::resize(new_size);
}

void AurynVectorFloat::scale(float a) 
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
	for ( NeuronID i = 0 ; i < size ; ++i ) {
		data[i] *= a;
	}
#endif /* CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY */
}


void AurynVectorFloat::saxpy(float a, AurynVectorFloat * x)
{
	check_size(x);
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

void AurynVectorFloat::mul(AurynVectorFloat * v) 
{
	check_size(v);
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


void AurynVectorFloat::add(float a) 
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	data[0:size:1] = a + data[0:size:1];
	#else
	const __m128 scalar = _mm_set1_ps(a);
	for ( float * i = data ; i != data+size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		// _mm_prefetch((i + SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS),  _MM_HINT_NTA);  
		__m128 chunk = sse_load( i );
		__m128 result = _mm_add_ps(chunk, scalar);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < size ; ++i ) {
		data[i] += a;
	}
#endif
}


void AurynVectorFloat::add(AurynVectorFloat * v) 
{
	check_size(v);
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	data[0:size:1] = data[0:size:1] + v->data[0:v->size:1];
	#else
	float * bd = v->data;
	for ( float * i = data ; i != data+size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk_a = sse_load( i );
		__m128 chunk_b = sse_load( bd ); bd+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_add_ps(chunk_a, chunk_b);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < size ; ++i ) {
		data[i] += v->data[i];
	}
#endif
}

void AurynVectorFloat::sum(AurynVectorFloat * a, AurynVectorFloat * b) 
{
	check_size(a);
	check_size(b);
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	float * ea = a->data;
	float * eb = b->data;
	for ( float * i = data ; i != data+size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk_a = sse_load( ea ); ea+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 chunk_b = sse_load( eb ); eb+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_add_ps(chunk_a, chunk_b);
		sse_store( i, result );
	}
#else
	AurynVector::sum(a,b);
#endif
}

void AurynVectorFloat::sum(AurynVectorFloat * a, const float b) 
{
	check_size(a);
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	float * ea = a->data;
	const __m128 scalar = _mm_set1_ps(b);
	for ( float * i = data ; i != data+size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk_a = sse_load( ea ); ea+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_add_ps(chunk_a, scalar);
		sse_store( i, result );
	}
#else
	AurynVector::sum(a,b);
#endif
}

void AurynVectorFloat::diff(AurynVectorFloat * a, AurynVectorFloat * b) 
{
	check_size(a);
	check_size(b);
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	float * ea = a->data;
	float * eb = b->data;
	for ( float * i = data ; i != data+size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		const __m128 chunk_a = sse_load( ea ); ea+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		const __m128 chunk_b = sse_load( eb ); eb+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_sub_ps(chunk_a, chunk_b);
		sse_store( i, result );
	}
#else
	AurynVector::diff(a,b);
#endif
}

void AurynVectorFloat::diff(AurynVectorFloat * a, const float b) 
{
	check_size(a);
	sum(a,-b);
}

void AurynVectorFloat::diff(const float a, AurynVectorFloat * b ) 
{
	check_size(b);
	sum(b,-a);
	neg();
}

void AurynVectorFloat::follow(AurynVectorFloat * v, const float rate)
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	for ( NeuronID i = 0 ; i < size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		const __m128 chunk_a = sse_load( v->data+i ); 
		const __m128 chunk_b = sse_load( data+i ); 
		const __m128 scalar  = _mm_set1_ps(rate);
		__m128 temp = _mm_sub_ps(chunk_a, chunk_b);
		temp = _mm_mul_ps( scalar, temp );
		temp = _mm_add_ps( chunk_b, temp );
		sse_store( data+i, temp );
	}
#else
	super::follow(v,rate);
#endif
}
