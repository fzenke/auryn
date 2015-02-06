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

#ifndef AURYN_DEFINITIONS_H_
#define AURYN_DEFINITIONS_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <cstdlib>
#include <list>
#include <exception>
#include <limits>


#ifndef CODE_ACTIVATE_CILK_INSTRUCTIONS
#include <x86intrin.h> // SIMD intrinsics
#else // XMM registers are not supported on the phi platform
#include <immintrin.h> // AVX only
#endif /* CODE_ACTIVATE_CILK_INSTRUCTIONS */

#include <boost/mpi.hpp>

#include <boost/archive/text_oarchive.hpp> 
#include <boost/archive/text_iarchive.hpp> 
#include <boost/archive/binary_oarchive.hpp> 
#include <boost/archive/binary_iarchive.hpp> 

#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

#include "Logger.h"

using namespace std;
namespace mpi = boost::mpi;


/*! The current Auryn version number */
#define VERSION 0.4

/*! Toggle between memory alignment for
 * SIMD code.
 */
#define CODE_ALIGNED_SIMD_INSTRUCTIONS

/*! Toggle prefetching in spike backpropagation */
#define CODE_ACTIVATE_PREFETCHING_INTRINSICS

/*! Toggle between using auryns vector 
 * operations using SIMD instructions. If
 * you do not enforce this here the compiler
 * might still choose to use them when
 * mtune settings are set appropriately.
 */
#define CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY

/*! Use Intel Cilk Plus -- only has an effect when 
 * CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY is enabled. */
// #define CODE_ACTIVATE_CILK_INSTRUCTIONS

#define SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS 4 //!< SSE can process 4 floats in parallel

// #define CODE_COLLECT_SYNC_TIMING_STATS //!< toggle  collection of timing data on sync/all_gather

/*! System wide integration time step */
const double dt = 1.0e-4;

/*! System wide minimum delay which determines
 * the sync interval between nodes in units of dt.
 */ 
#define MINDELAY 8

/*! Groups with an effective size smaller than
 *  this will not be distributed. Furthermore 
 *  groups that are distributed will not be 
 *  cut up in chunks that are not smaller than
 *  this. This is done to reduce overhead
 */
#define DEFAULT_MINDISTRIBUTEDSIZE 16


/*! These precompiler directives control
 * what type of synaptic traces Auryn
 * implements for STDP models.
 */
#define DEFAULT_TRACE_MODEL EulerTrace
#define PRE_TRACE_MODEL EulerTrace
// To switch to LinearTrace as default 
// pre_trace model uncomment the following 
// lines:
// #define PRE_TRACE_MODEL_LINTRACE dummy
// #undef  PRE_TRACE_MODEL
// #define PRE_TRACE_MODEL LinearTrace

/*! Specifies the different transmitter types 
 * that Auryn knows. */
enum TransmitterType { 
	GLUT, //!< Standard Glutamatergic (excitatory) transmission.
	GABA, //!< Standard Gabaergic (inhibitory) transmission.
	AMPA, //!< Only targets AMPA channels.
	NMDA, //!< Only targets NMDA.
	MEM   //!< Current based synapse. Adds the transmitted quantity directly to membrane voltage.
};

enum StimulusGroupModeType { MANUAL, RANDOM, SEQUENTIAL, SEQUENTIAL_REV, STIMFILE };


typedef unsigned int NeuronID; //!< NeuronID is an unsigned integeger type used to index neurons in Auryn.
typedef NeuronID AurynInt;
typedef unsigned int StateID; //!< StateID is an unsigned integeger type used to index synaptic states in Auryn.
typedef unsigned long AurynLong; //!< An unsigned long type used to count synapses or similar. 
typedef NeuronID AurynTime; //!< Defines Auryns discrete time unit of the System clock.  Change to AurynLong if 120h of simtime are not sufficient
typedef float AurynFloat; //!< Low precision floating point datatype.
typedef double AurynDouble; //!< Higher precision floating point datatype.
typedef AurynFloat AurynWeight; //!< Unit of synaptic weights.
typedef AurynFloat AurynState; //!< Type for Auryn state variables (default single precision since it needs to be compatible with auryn_vector_float).
typedef vector<NeuronID> SpikeContainer; //!< Spike container type. Used for storing spikes.
typedef vector<float> AttributeContainer; //!< Attribute container type. Used for storing spike attributes that are needed for efficient STP implementations.


// Auryn vector template -- copies the core of GSL vector functionality
template <typename T> 
struct auryn_vector { 
    NeuronID size;
    T * data;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & size;
		for ( NeuronID i = 0 ; i < size ; ++i ) 
			ar & data[i];
	}
};

/*! Reimplements a simplified version of the GSL vector. */
typedef auryn_vector<AurynFloat> auryn_vector_float;

typedef auryn_vector<unsigned short> auryn_vector_ushort;

struct neuron_pair {
	NeuronID i,j;
} ;

struct pattern_member {
	NeuronID i;
	AurynDouble gamma;
} ;
typedef vector<pattern_member> type_pattern;



/*! Determines memory alignment (adapted from ATLAS library) 
 @param N max return value
 @param *vp Pointer to be aligned 
 @param inc size of element, in bytes
 @param align required alignment, in bytes */
int auryn_AlignOffset (const int N, const void *vp, const int inc, const int align);  

/*! Rounds vector size to multiple of four to allow using the SSE optimizations. */
NeuronID calculate_vector_size(NeuronID i);


// Float vector functions

/*! Allocates an auryn_vector_float */
auryn_vector_float * auryn_vector_float_alloc(const NeuronID n);
/*! Frees an auryn_vector_float */
void auryn_vector_float_free (auryn_vector_float * v);
/*! Initializes an auryn_vector_float with zeros */
void auryn_vector_float_set_zero (auryn_vector_float * v);
/*! Sets all elements in an auryn_vector_float to value x */
void auryn_vector_float_set_all (auryn_vector_float * v, AurynFloat x);
/*! Copies vector src to dst assuming they have the same size. 
 * Otherwise this will lead to undefined results. No checking of size is
 * performed for performance reasons. */
void auryn_vector_float_copy (auryn_vector_float * src, auryn_vector_float * dst );
/*! Auryn vector getter */
AurynFloat auryn_vector_float_get (const auryn_vector_float * v, const NeuronID i);
/*! Auryn vector setter */
void auryn_vector_float_set (auryn_vector_float * v, const NeuronID i, AurynFloat x);
/*! Auryn vector gets pointer to designed element. */
AurynFloat * auryn_vector_float_ptr (const auryn_vector_float * v, const NeuronID i);

/*! Internal  version of auryn_vector_float_mul of gsl operations */
void auryn_vector_float_mul( auryn_vector_float * a, auryn_vector_float * b);
/*! Internal  version of auryn_vector_float_add gsl operations */
void auryn_vector_float_add_constant( auryn_vector_float * a, float b );
/*! Internal  SAXPY version */
void auryn_vector_float_saxpy( const float a, const auryn_vector_float * x, const auryn_vector_float * y );
/*! Internal  version to scale a vector with a constant b  */
void auryn_vector_float_scale(const float a, const auryn_vector_float * b );
/*! Internal  version to clip all the elements of a vector between [a:b]  */
void auryn_vector_float_clip(auryn_vector_float * v, const float a , const float b );
/*! Internal  version to clip all the elements of a vector between [a:0]  */
void auryn_vector_float_clip(auryn_vector_float * v, const float a );
/*! Internal  version of to add GSL vectors */
void auryn_vector_float_add( auryn_vector_float * a, auryn_vector_float * b);
/*! Internal  version of to subtract GSL vectors */
void auryn_vector_float_sub( auryn_vector_float * a, auryn_vector_float * b);


// ushort vector functions
/*! Allocates an auryn_vector_ushort */
auryn_vector_ushort * auryn_vector_ushort_alloc(const NeuronID n);
/*! Frees an auryn_vector_ushort */
void auryn_vector_ushort_free (auryn_vector_ushort * v);
/*! Initializes an auryn_vector_ushort with zeros */
void auryn_vector_ushort_set_zero (auryn_vector_ushort * v);
/*! Sets all elements in an auryn_vector_ushort to value x */
void auryn_vector_ushort_set_all (auryn_vector_ushort * v, unsigned short x);
/*! Copies vector src to dst assuming they have the same size. 
 * Otherwise this will lead to undefined results. No checking of size is
 * performed for performance reasons. */
void auryn_vector_ushort_copy (auryn_vector_ushort * src, auryn_vector_ushort * dst );
/*! Auryn vector getter */
unsigned short auryn_vector_ushort_get (const auryn_vector_ushort * v, const NeuronID i);
/*! Auryn vector setter */
void auryn_vector_ushort_set (auryn_vector_ushort * v, const NeuronID i, unsigned short x);
/*! Auryn vector gets pointer to designed element. */
unsigned short * auryn_vector_ushort_ptr (const auryn_vector_ushort * v, const NeuronID i);
/*! Auryn spike event for binary monitors */
struct spikeEvent_type
{
    AurynTime time;
    NeuronID neuronID;
};

// Exceptions
class AurynOpenFileException: public exception
{
	  virtual const char* what() const throw()
		    {
				    return "Failed opening file.";
		    }
};

class AurynMMFileException: public exception
{
	  virtual const char* what() const throw()
		    {
				    return "Problem reading MatrixMarket file. Not row major format?";
		    }
};

class AurynMatrixDimensionalityException: public exception
{
	  virtual const char* what() const throw()
		    {
				    return "Cannot add data outside of matrix.";
		    }
};

class AurynMatrixBufferException: public exception
{
	  virtual const char* what() const throw()
		    {
				    return "Buffer full.";
		    }
};

class AurynMatrixPushBackException: public exception
{
	  virtual const char* what() const throw()
		    {
				    return "Could not push_back in SimpleMatrix. Out of order execution?";
		    }
};

class AurynConnectionAllocationException: public exception
{
	  virtual const char* what() const throw()
		    {
				    return "Buffer has not been allocated.";
		    }
};


class AurynMemoryAlignmentException: public exception
{
	  virtual const char* what() const throw()
		    {
				    return "Memory not aligned to 16bytes.";
		    }
};

class AurynDelayTooSmallException: public exception
{
	  virtual const char* what() const throw()
		    {
				    return "One of the SpikeDelays was chosen shorter than the current value of MINDELAY.";
		    }
};

class AurynTimeOverFlowException: public exception
{
	  virtual const char* what() const throw()
		    {
				    return "Trying to simulate more timesteps than are available in AurynTime.";
		    }
};


#endif /*AURYN_DEFINITIONS_H__*/
