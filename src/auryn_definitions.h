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
#include <x86intrin.h> // SIMD intrinsics (pulls everything you need)
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

namespace mpi = boost::mpi;

/*! The current Auryn version/revision number. 
 * Should be all ints. */
#define AURYNVERSION 0
#define AURYNSUBVERSION 8
#define AURYNREVISION 0
#define AURYNVERSIONSUFFIX "-dev"


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


namespace auryn {

	/*! \brief Simulator wide integration time step */
	const double dt = 1.0e-4;

	/*! \brief Specifies the different transmitter types 
	 * that Auryn knows. */
	enum TransmitterType { 
		GLUT, //!< Standard Glutamatergic (excitatory) transmission.
		GABA, //!< Standard Gabaergic (inhibitory) transmission.
		AMPA, //!< Only targets AMPA channels.
		NMDA, //!< Only targets NMDA.
		MEM,   //!< Current based synapse. Adds the transmitted quantity directly to membrane voltage.
		CURSYN   //!< Current based synapse with dynamics.
	};

	/*! \brief Specifies stimulus order used in StimulusGroup */
	enum StimulusGroupModeType { MANUAL, RANDOM, SEQUENTIAL, SEQUENTIAL_REV, STIMFILE };


	typedef unsigned int NeuronID; //!< NeuronID is an unsigned integeger type used to index neurons in Auryn.
	typedef NeuronID AurynInt;
	typedef unsigned int StateID; //!< StateID is an unsigned integeger type used to index synaptic states in Auryn.
	typedef unsigned long AurynLong; //!< An unsigned long type used to count synapses or similar. 
	typedef NeuronID AurynTime; //!< Defines Auryns discrete time unit of the System clock.  Change to AurynLong if 120h of simtime are not sufficient
	typedef std::string string; //!< Standard library string type which is imported into Auryn namespace.
	typedef float AurynFloat; //!< Low precision floating point datatype.
	typedef double AurynDouble; //!< Higher precision floating point datatype.
	typedef AurynFloat AurynWeight; //!< Unit of synaptic weights.
	typedef AurynFloat AurynState; //!< Type for Auryn state variables (default single precision since it needs to be compatible with auryn_vector_float).
	typedef std::vector<NeuronID> SpikeContainer; //!< Spike container type. Used for storing spikes.
	typedef std::vector<float> AttributeContainer; //!< Attribute container type. Used for storing spike attributes that are needed for efficient STP implementations.



	/*! \brief Struct that defines a pair of neurons in SparseConnection */
	struct neuron_pair {
		NeuronID i,j;
	} ;

	/*! \brief Struct used to define neuronal assembly patterns in SparseConnection */
	struct pattern_member {
		NeuronID i;
		AurynDouble gamma;
	} ;

	typedef std::vector<pattern_member> type_pattern;



	/*! \brief Determines memory alignment (adapted from ATLAS library) 
	 *
	 @param N max return value
	 @param *vp Pointer to be aligned 
	 @param inc size of element, in bytes
	 @param align required alignment, in bytes */
	int auryn_AlignOffset (const int N, const void *vp, const int inc, const int align);  

	/*! \brief Rounds vector size to multiple of four to allow using the SSE optimizations. */
	NeuronID calculate_vector_size(NeuronID i);


	/*! \brief Auryn spike event for binary monitors */
	struct SpikeEvent_type
	{
		AurynTime time; 
		NeuronID neuronID;
	};

	/*! \brief Tag for header in binary encoded spike monitor files. 
	 *
	 * The first digits are 28796 for Auryn in 
	 * phone dial notation. The remaining 4 digits encode type of binary file and the current Auryn 
	 * version */
	const NeuronID tag_binary_spike_monitor = 287960000+100*AURYNVERSION+10*AURYNSUBVERSION+1*AURYNREVISION;
	const NeuronID tag_binary_state_monitor = 287961000+100*AURYNVERSION+10*AURYNSUBVERSION+1*AURYNREVISION;

	// Exceptions
	class AurynOpenFileException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Failed opening file.";
				}
	};

	class AurynMMFileException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Problem reading MatrixMarket file. Not row major format?";
				}
	};

	class AurynMatrixDimensionalityException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Cannot add data outside of matrix.";
				}
	};

	class AurynMatrixComplexStateException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Trying to access non existing complex synaptic states.";
				}
	};

	class AurynMatrixBufferException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Buffer full.";
				}
	};

	class AurynMatrixPushBackException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Could not push_back in SimpleMatrix. Out of order execution?";
				}
	};

	class AurynConnectionAllocationException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Buffer has not been allocated.";
				}
	};


	class AurynMemoryAlignmentException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Memory not aligned or problem allocating aligned memory.";
				}
	};

	class AurynDelayTooSmallException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "One of the SpikeDelays was chosen shorter than the current value of MINDELAY.";
				}
	};

	class AurynTimeOverFlowException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Trying to simulate more timesteps than are available in AurynTime.";
				}
	};

	class AurynStateVectorException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Auryn encountered an undefined problem when dealing with StateVectors.";
				}
	};

	class AurynGenericException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Auryn encountered a problem which it deemed serious enough to break the run. \
							To debug set logger vebosity to VERBOSE or EVERYTHING and analyze the log files.";
				}
	};

	class AurynVectorDimensionalityException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "Dimensions do not match or trying to read beyond vector size. " 
							   "Are the vectors zero padded to a multiples of four dimension?";
				}
	};

	class AurynSpikeAttributeSizeException: public std::exception
	{
		  virtual const char* what() const throw()
				{
						return "The number of spike attributes can only be increased.";
				}
	};



	// forward declartion of template class which is implemented in AurynVector.h
	template <typename T, typename IndexType > 
	class AurynVector;

	class AurynVectorFloat; // Forward declaration

	/*! \brief Defines AurynStateVector type as synonymous to AurynVectorFloat
	 *
	 * Auryn state vectors are used to implement vectorized code for SpikingGroup and NeuronGroup.
	 * An AurynStateVector in a SpikingGroup typically has the local rank size of that group and 
	 * each neuronal state variable corresponds to a state vector that houses this state for all neurons on that rank.
	 * AurynStateVectors are defined as AurynVectorFloat. 
	 * This typically needs to change when AurynState or AurynFloat types are changed. */
	typedef AurynVectorFloat AurynStateVector; 

	/*! \brief Defines AurynSynStateVector for synaptic states */
	typedef AurynVector<AurynWeight, AurynLong> AurynSynStateVector; 

	// Legacy state vector types 
	/*! \brief Legacy definition of AurynStateVector */
	typedef AurynStateVector auryn_vector_float; //!< Default legacy Auryn state vector type

	/*! \brief Legacy definition of AurynVector<unsigned short> */
	typedef AurynVector<unsigned short, NeuronID> auryn_vector_ushort; //!< Default legacy Auryn ushort vector type


	// Legacy float vector functions
	// These functions should not be used any more in the future. Instead use the member functions of AurynVectorFloat

	/*! Allocates an auryn_vector_float */
	auryn_vector_float * auryn_vector_float_alloc(const NeuronID n);

	/*! Frees an auryn_vector_float */
	void auryn_vector_float_free (auryn_vector_float * v);

	/*! Initializes an auryn_vector_float with zeros */
	void auryn_vector_float_set_zero (auryn_vector_float * v);

	/*! Sets all elements in an auryn_vector_float to value x */
	void auryn_vector_float_set_all (auryn_vector_float * v, AurynFloat x);

	/*! \brief Copies vector src to dst assuming they have the same size. 
	 *
	 * Otherwise this will lead to undefined results. No checking of size is
	 * performed for performance reasons. */
	void auryn_vector_float_copy (auryn_vector_float * src, auryn_vector_float * dst );

	/*! Auryn vector getter */
	AurynFloat auryn_vector_float_get (auryn_vector_float * v, const NeuronID i);

	/*! Auryn vector setter */
	void auryn_vector_float_set (auryn_vector_float * v, const NeuronID i, AurynFloat x);

	/*! Auryn vector gets pointer to designed element. */
	AurynFloat * auryn_vector_float_ptr (auryn_vector_float * v, const NeuronID i);

	/*! Internal  version of auryn_vector_float_mul of gsl operations */
	void auryn_vector_float_mul( auryn_vector_float * a, auryn_vector_float * b);

	/*! \brief Computes a := a + b 
	 *
	 * Internal  version of auryn_vector_float_add between a constant and a vector */
	void auryn_vector_float_add_constant( auryn_vector_float * a, float b );

	/*! Computes y := a*x+y
	 *
	 * Internal SAXPY version */
	void auryn_vector_float_saxpy( const float a, auryn_vector_float * x, auryn_vector_float * y );
	/*! Internal version to scale a vector with a constant b  */
	void auryn_vector_float_scale(const float a, auryn_vector_float * b );
	/*! Internal version to clip all the elements of a vector between [a:b]  */
	void auryn_vector_float_clip(auryn_vector_float * v, const float a , const float b );

	/*! Internal  version to clip all the elements of a vector between [a:0]  */
	void auryn_vector_float_clip(auryn_vector_float * v, const float a );

	/*! \brief Internal  version of to add GSL vectors.
	 *
	 * Add vectors a and b and store the result in a. */
	void auryn_vector_float_add( auryn_vector_float * a, auryn_vector_float * b);

	/*! \brief Computes a := a-b
	 * 
	 *  Internal  version of to subtract GSL vectors. */
	void auryn_vector_float_sub( auryn_vector_float * a, auryn_vector_float * b);

	/*! \brief Computes r := a-b */
	void auryn_vector_float_sub( auryn_vector_float * a, auryn_vector_float * b, auryn_vector_float * r);



	// Legacy ushort vector functions
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
	unsigned short auryn_vector_ushort_get (auryn_vector_ushort * v, const NeuronID i);
	/*! Auryn vector setter */
	void auryn_vector_ushort_set (auryn_vector_ushort * v, const NeuronID i, unsigned short x);
	/*! Auryn vector gets pointer to designed element. */
	unsigned short * auryn_vector_ushort_ptr (auryn_vector_ushort * v, const NeuronID i);

} // namespace

#endif /*AURYN_DEFINITIONS_H__*/
