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

#include "auryn_definitions.h"

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
	return i+(SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS-i%SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS);
}

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

void auryn_vector_float_mul( gsl_vector_float * a, gsl_vector_float * b)
{
#ifdef USE_SIMD_INSTRUCTIONS_EXPLICITLY
	float * bd = b->data;
	for ( float * i = a->data ; i != a->data+a->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk_a = sse_load( i );
		__m128 chunk_b = sse_load( bd ); bd+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_mul_ps(chunk_a, chunk_b);
		sse_store( i, result );
	}
#else
	gsl_vector_float_mul( a, b );
#endif
}

void auryn_vector_float_add_constant( gsl_vector_float * a, const float b )
{
#ifdef USE_SIMD_INSTRUCTIONS_EXPLICITLY
	const __m128 scalar = _mm_set1_ps(b);
	for ( float * i = a->data ; i != a->data+a->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		// _mm_prefetch((i + SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS),  _MM_HINT_NTA);  
		__m128 chunk = sse_load( i );
		__m128 result = _mm_add_ps(chunk, scalar);
		sse_store( i, result );
	}
#else
	for ( float * i = a->data ; i != a->data+a->size ; ++i ) {
		*i += b;
	}
#endif
}

void auryn_vector_float_scale( const float a, const gsl_vector_float * b )
{
#ifdef USE_SIMD_INSTRUCTIONS_EXPLICITLY
	const __m128 scalar = _mm_set1_ps(a);
	for ( float * i = b->data ; i != b->data+b->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( i );
		__m128 result = _mm_mul_ps(chunk, scalar);
		sse_store( i, result );
	}
#else
	for ( float * i = b->data ; i < b->data+b->size ; ++i ) {
		*i *= a;
	}
#endif
}

void auryn_vector_float_saxpy( const float a, const gsl_vector_float * x, const gsl_vector_float * y )
{
	float * xp = x->data;
#ifdef USE_SIMD_INSTRUCTIONS_EXPLICITLY
	const __m128 alpha = _mm_set1_ps(a);
	for ( float * i = y->data ; i < y->data+y->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( xp ); xp += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result     = _mm_mul_ps( alpha, chunk );

		chunk  = sse_load( i );
		result = _mm_add_ps( result, chunk );
		sse_store( i, result ); 
	}
#else
	for ( float * i = y->data ; i < y->data+y->size ; ++i ) {
		*i = a * *xp + *i; ++xp;
	}
#endif
}

void auryn_vector_float_add( gsl_vector_float * a, gsl_vector_float * b)
{
#ifdef USE_SIMD_INSTRUCTIONS_EXPLICITLY
	float * bd = b->data;
	for ( float * i = a->data ; i != a->data+a->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk_a = sse_load( i );
		__m128 chunk_b = sse_load( bd ); bd+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_add_ps(chunk_a, chunk_b);
		sse_store( i, result );
	}
#else
	gsl_vector_float_add( a, b );
#endif
}

void auryn_vector_float_sub( gsl_vector_float * a, gsl_vector_float * b)
{
#ifdef USE_SIMD_INSTRUCTIONS_EXPLICITLY
	float * bd = b->data;
	for ( float * i = a->data ; i != a->data+a->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk_a = sse_load( i );
		__m128 chunk_b = sse_load( bd ); bd+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_sub_ps(chunk_a, chunk_b);
		sse_store( i, result );
	}
#else
	gsl_vector_float_sub( a, b );
#endif
}

void auryn_vector_float_clip( gsl_vector_float * v, const float a, const float b ) {
#ifdef USE_SIMD_INSTRUCTIONS_EXPLICITLY
	const __m128 lo = _mm_set1_ps(a);
	const __m128 hi = _mm_set1_ps(b);
	for ( float * i = v->data ; i != v->data+v->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( i );
		__m128 result = _mm_min_ps(chunk, hi);
		result = _mm_max_ps(result, lo);
		sse_store( i, result );
	}
#else
	for ( float * i = v->data ; i != v->data+v->size ; ++i ) {
		if ( *i < a ) {
			*i = a;
		} else 
			if ( *i > b ) 
				*i = b;
	}
#endif
}

void auryn_vector_float_clip( gsl_vector_float * v, const float a ) {
#ifdef USE_SIMD_INSTRUCTIONS_EXPLICITLY
	const __m128 lo = _mm_set1_ps(a);
	const __m128 hi = _mm_set1_ps(0.);
	for ( float * i = v->data ; i != v->data+v->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( i );
		__m128 result = _mm_min_ps(chunk, hi);
		result = _mm_max_ps(result, lo);
		sse_store( i, result );
	}
#else
	auryn_vector_float_clip( v, a, 1e127 );
#endif
}
