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

#ifndef AURYNVECTOR_H_
#define AURYNVECTOR_H_

#include <boost/mpi.hpp>

#include <boost/archive/text_oarchive.hpp> 
#include <boost/archive/text_iarchive.hpp> 
#include <boost/archive/binary_oarchive.hpp> 
#include <boost/archive/binary_iarchive.hpp> 

#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>


#include "auryn_definitions.h"


namespace auryn {

	/*! \brief Auryn vector template 
	 *
	 * Copies the core of GSL vector functionality 
	 * */
	template <typename T> 
	class AurynVector { 
		private: 
			friend class boost::serialization::access;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & size;
				for ( NeuronID i = 0 ; i < size ; ++i ) 
					ar & data[i];
			}


		public: 
			NeuronID size;
			T * data;

			/*! \brief Default constructor */
			AurynVector(NeuronID n) 
			{
				T * data = new T [n];
				size = n;
				data = data;
			}

			/*! \brief Default destructor */
			~AurynVector() 
			{
				delete data;
			}

			/*! \brief Set all elements to value v. */
			void set_all(T v) 
			{
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					data[i] = v;
				}
			}

			/*! \brief Set all elements to zero. */
			void set_zero(T v) 
			{
				set_all(0.0);
			}

			/*! \brief Scales all vector elements by a. */
			void scale(AurynFloat a) 
			{
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					data[i] *= a;
				}
			}

	};

	typedef AurynVector<AurynFloat> auryn_vector_float; //!< Reimplements a simplified version of the GSL vector.

	typedef AurynVector<unsigned short> auryn_vector_ushort; //!< Reimplements a simplified version of the GSL vector for ushort.

	// Float vector legacy functions

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
	AurynFloat auryn_vector_float_get (const auryn_vector_float * v, const NeuronID i);

	/*! Auryn vector setter */
	void auryn_vector_float_set (auryn_vector_float * v, const NeuronID i, AurynFloat x);

	/*! Auryn vector gets pointer to designed element. */
	AurynFloat * auryn_vector_float_ptr (const auryn_vector_float * v, const NeuronID i);

	/*! Internal  version of auryn_vector_float_mul of gsl operations */
	void auryn_vector_float_mul( auryn_vector_float * a, auryn_vector_float * b);

	/*! \brief Computes a := a + b 
	 *
	 * Internal  version of auryn_vector_float_add between a constant and a vector */
	void auryn_vector_float_add_constant( auryn_vector_float * a, float b );

	/*! Computes y := a*x+y
	 *
	 * Internal SAXPY version */
	void auryn_vector_float_saxpy( const float a, const auryn_vector_float * x, const auryn_vector_float * y );
	/*! Internal version to scale a vector with a constant b  */
	void auryn_vector_float_scale(const float a, const auryn_vector_float * b );
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

}


#endif /*AURYNVECTOR_H_*/
