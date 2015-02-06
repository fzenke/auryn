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

void auryn_vector_float_mul( auryn_vector_float * a, auryn_vector_float * b)
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	a->data[0:a->size:1] = a->data[0:a->size:1] * b->data[0:b->size:1];
	#else
	float * bd = b->data;
	for ( float * i = a->data ; i != a->data+a->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk_a = sse_load( i );
		__m128 chunk_b = sse_load( bd ); bd+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_mul_ps(chunk_a, chunk_b);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < a->size ; ++i ) {
		a->data[i] *= b->data[i];
	}
#endif
}

void auryn_vector_float_add_constant( auryn_vector_float * a, const float b )
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	a->data[0:a->size:1] = b + a->data[0:a->size:1];
	#else
	const __m128 scalar = _mm_set1_ps(b);
	for ( float * i = a->data ; i != a->data+a->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		// _mm_prefetch((i + SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS),  _MM_HINT_NTA);  
		__m128 chunk = sse_load( i );
		__m128 result = _mm_add_ps(chunk, scalar);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < a->size ; ++i ) {
		a->data[i] += b;
	}
#endif
}

void auryn_vector_float_scale( const float a, const auryn_vector_float * b )
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	b->data[0:b->size:1] = a * b->data[0:b->size:1];
	#else
	const __m128 scalar = _mm_set1_ps(a);
	for ( float * i = b->data ; i != b->data+b->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( i );
		__m128 result = _mm_mul_ps(chunk, scalar);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < b->size ; ++i ) {
		b->data[i] *= a;
	}
#endif /* CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY */
}

void auryn_vector_float_saxpy( const float a, const auryn_vector_float * x, const auryn_vector_float * y )
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	y->data[0:y->size:1] = a * x->data[0:x->size:1] + y->data[0:y->size:1];
	#else
	float * xp = x->data;
	const __m128 alpha = _mm_set1_ps(a);
	for ( float * i = y->data ; i < y->data+y->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( xp ); xp += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result     = _mm_mul_ps( alpha, chunk );

		chunk  = sse_load( i );
		result = _mm_add_ps( result, chunk );
		sse_store( i, result ); 
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < y->size ; ++i ) {
		y->data[i] += a * x->data[i];
	}
#endif
}

void auryn_vector_float_add( auryn_vector_float * a, auryn_vector_float * b)
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	a->data[0:a->size:1] = a->data[0:a->size:1] + b->data[0:b->size:1];
	#else
	float * bd = b->data;
	for ( float * i = a->data ; i != a->data+a->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk_a = sse_load( i );
		__m128 chunk_b = sse_load( bd ); bd+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_add_ps(chunk_a, chunk_b);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < a->size ; ++i ) {
		a->data[i] += b->data[i];
	}
#endif
}

void auryn_vector_float_sub( auryn_vector_float * a, auryn_vector_float * b)
{
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	a->data[0:a->size:1] = a->data[0:a->size:1] - b->data[0:b->size:1];
	#else
	float * bd = b->data;
	for ( float * i = a->data ; i != a->data+a->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk_a = sse_load( i );
		__m128 chunk_b = sse_load( bd ); bd+=SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
		__m128 result = _mm_sub_ps(chunk_a, chunk_b);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < a->size ; ++i ) {
		a->data[i] -= b->data[i];
	}
#endif
}

void auryn_vector_float_clip( auryn_vector_float * v, const float a, const float b ) {
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	for ( NeuronID i = 0 ; i < v->size ; ++i ) {
		if ( v->data[i] < a ) {
			v->data[i] = a;
		} else 
			if ( v->data[i] > b ) 
				v->data[i] = b;
	}
	#else
	const __m128 lo = _mm_set1_ps(a);
	const __m128 hi = _mm_set1_ps(b);
	for ( float * i = v->data ; i != v->data+v->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( i );
		__m128 result = _mm_min_ps(chunk, hi);
		result = _mm_max_ps(result, lo);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	for ( NeuronID i = 0 ; i < v->size ; ++i ) {
		if ( v->data[i] < a ) {
			v->data[i] = a;
		} else 
			if ( v->data[i] > b ) 
				v->data[i] = b;
	}
#endif
}

void auryn_vector_float_clip( auryn_vector_float * v, const float a ) {
#ifdef CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY
	#ifdef CODE_ACTIVATE_CILK_INSTRUCTIONS
	auryn_vector_float_clip( v, a, 1e16 );
	#else
	const __m128 lo = _mm_set1_ps(a);
	const __m128 hi = _mm_set1_ps(0.);
	for ( float * i = v->data ; i != v->data+v->size ; i += SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS )
	{
		__m128 chunk = sse_load( i );
		__m128 result = _mm_min_ps(chunk, hi);
		result = _mm_max_ps(result, lo);
		sse_store( i, result );
	}
	#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */
#else
	auryn_vector_float_clip( v, a, 1e16 );
#endif
}

auryn_vector_float * auryn_vector_float_alloc( const NeuronID n ) {
	AurynFloat * data = new AurynFloat [n];
	auryn_vector_float * vec = new auryn_vector_float();
	vec->size = n;
	vec->data = data;
	return vec;
}

void auryn_vector_float_free ( auryn_vector_float * v ) {
	delete [] v->data;
	delete v;
}

void auryn_vector_float_set_all ( auryn_vector_float * v, AurynFloat x ) {
	for ( NeuronID i = 0 ; i < v->size ; ++ i ) 
		v->data[i] = x;
}

void auryn_vector_float_set_zero ( auryn_vector_float * v ) {
	auryn_vector_float_set_all(v, 0.0);
}

AurynFloat auryn_vector_float_get ( const auryn_vector_float * v, const NeuronID i ) {
	return v->data[i];
}

AurynFloat * auryn_vector_float_ptr ( const auryn_vector_float * v, const NeuronID i ) {
	return v->data+i;
}

void auryn_vector_float_set ( auryn_vector_float * v, const NeuronID i, AurynFloat x ) {
	v->data[i] = x;
}

void auryn_vector_float_copy ( auryn_vector_float * src, auryn_vector_float * dst ) {
	for ( NeuronID i = 0 ; i < dst->size ; ++i ) 
		dst->data[i] = src->data[i];
}


auryn_vector_ushort * auryn_vector_ushort_alloc( const NeuronID n ) {
	unsigned short * data = new unsigned short [n];
	auryn_vector_ushort * vec = new auryn_vector_ushort();
	vec->size = n;
	vec->data = data;
	return vec;
}

void auryn_vector_ushort_free ( auryn_vector_ushort * v ) {
	delete [] v->data;
	delete v;
}

void auryn_vector_ushort_set_all ( auryn_vector_ushort * v, unsigned short x ) {
	for ( NeuronID i = 0 ; i < v->size ; ++ i ) 
		v->data[i] = x;
}

void auryn_vector_ushort_set_zero ( auryn_vector_ushort * v ) {
	auryn_vector_ushort_set_all(v, 0);
}

unsigned short auryn_vector_ushort_get ( const auryn_vector_ushort * v, const NeuronID i ) {
	return v->data[i];
}

unsigned short * auryn_vector_ushort_ptr ( const auryn_vector_ushort * v, const NeuronID i ) {
	return v->data+i;
}

void auryn_vector_ushort_set ( auryn_vector_ushort * v, const NeuronID i, unsigned short x ) {
	v->data[i] = x;
}

void auryn_vector_ushort_copy ( auryn_vector_ushort * src, auryn_vector_ushort * dst ) {
	for ( NeuronID i = 0 ; i < dst->size ; ++i ) 
		dst->data[i] = src->data[i];
}

