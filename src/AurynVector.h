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

#include "auryn_definitions.h"


namespace auryn {

	/*! \brief Default Auryn vector template 
	 *
	 * This essentially copies the core of GSL vector functionality and makes it a class for easier handling.
	 * For performance reasons, time critical functions of this template have to be reimplemented in derived 
	 * classes with a specific template parameter T. For instance I will always provide a derived type AurynVectorFloat 
	 * which will per default be synonymous to AurynStateVector which implements SSE instructions for labour 
	 * intensive operations on the vectors. 
	 * Note, that all Auryn vectors should initialized with multiple of 4 elements (later that number might change) when
	 * we add AVX support to the code. If you use get_vector_size functions from SpikingGroup this will automatically
	 * be taken care of...
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

		protected:
			/*! \brief Checks if argument is larger than size and throws and exception if so 
			 *
			 * Check only enabled if NDEBUG is not defined.*/
			virtual void check_size(NeuronID x)
			{
#ifndef NDEBUG
				if ( x >= size ) {
					throw AurynVectorDimensionalityException();
				}
#endif 
			}

			/*! \brief Checks if vector size matches to this instance
			 *
			 * Check only enabled if NDEBUG is not defined.*/
			virtual void check_size(AurynVector * v)
			{
#ifndef NDEBUG
				if ( v->size != size ) {
					throw AurynVectorDimensionalityException();
				}
#endif 
			}

		public:
			// We keep these params public for legacy compatibility reasons
			NeuronID size;
			T * data __attribute__((aligned(16)));

			/*! \brief Default constructor */
			AurynVector(NeuronID n) 
			{
				size = n;
				data = new T [n];
				set_zero(); // let's give it a defined initial value
			}

			/*! \brief Default destructor */
			virtual ~AurynVector() 
			{
				delete data;
			}

			/*! \brief Set all elements to value v. */
			virtual void set_all(T v) 
			{
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					data[i] = v;
				}
			}

			/*! \brief Set all elements to zero. */
			virtual void set_zero() 
			{
				set_all(0.0);
			}

			/*! \brief Scales all vector elements by a. */
			virtual void scale(AurynFloat a) 
			{
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					data[i] *= a;
				}
			}

			/*! \brief Adds constant c to each vector element */
			virtual void add(AurynFloat c) 
			{
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					data[i] += c;
				}
			}

			/*! \brief Adds the value c to specific vector element i */
			virtual void add_specific(NeuronID i, AurynFloat c) 
			{
				check_size(i);
				data[i] += c;
			}

			/*! \brief Adds a vector v to the vector
			 *
			 * No checking of the dimensions match! */
			virtual void add(AurynVector * v) 
			{
				check_size(v);
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					data[i] += v->data[i];
				}
			}

			/*! \brief Subtract constant c to each vector element */
			virtual void sub(AurynFloat c) 
			{
				add(-c);
			}

			/*! \brief Elementwise subtraction */
			virtual void sub(AurynVector * v) 
			{
				check_size(v);
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					data[i] -= v->data[i];
				}
			}

			/*! \brief Multiply all vector elements by constant */
			virtual void mul(AurynFloat a) 
			{
				scale(a);
			}

			/*! \brief Element-wise vector multiply  
			 *
			 * */
			virtual void mul(AurynVector * v) 
			{
				check_size(v);
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					data[i] *= v->data[i];
				}
			}

			/*! \brief Copies vector v 
			 *
			 * */
			virtual void copy(AurynVector * v) 
			{
				check_size(v);
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					data[i] = v->data[i];
				}
			}


			/*! \brief SAXPY operation as in GSL 
			 *
			 * Computes a*x + y and stores the result to y where y is the present instance. 
			 * \param a The scaling factor for the additional vector
			 * \param x The additional vector to add
			 * */
			virtual void saxpy(AurynFloat a, AurynVector * x) 
			{
				check_size(x);
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					data[i] += a * x->data[i];
				}
			}

			/*! \brief Gets element i from vector */
			T get(NeuronID i)
			{
				check_size(i);
				return data[i];
			}

			/*! \brief Gets pointer to element i from vector 
			 *
			 * When no argument is given the function returns the first element of 
			 * data array of the vector. */
			T * ptr(NeuronID i = 0)
			{
				check_size(i);
				return data+i;
			}

			/*! \brief Sets element i in vector to value */
			void set(NeuronID i, T value)
			{
				check_size(i);
				data[i] = value;
			}

			/*! \brief Clips all vector elements to the range min max
			 *
			 * \param min Minimum value
			 * \param max Maximum value
			 */
			void clip(T min, T max)
			{
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					if ( data[i] < min ) {
						 data[i] = min;
					} else 
						if ( data[i] > max ) 
							 data[i] = max;
				}
			}

			/*! \brief Print vector elements to std out for debugging */
			void print() {
				for ( NeuronID i = 0 ; i < size ; ++i ) {
					std::cout << get(i) << " ";
				}
				std::cout << std::endl;
			}
	};

	/*! \brief Derived AurynVectorFloat class for performance computation
	 *
	 * This class inherits the template AurynVector<float> and overwrites 
	 * some of the functions defined in the template with SIMD intrinsics 
	 * for higher performance.
	 */
	class AurynVectorFloat : public AurynVector<float> 
	{

		public:
			/*! \brief Default constructor */
			AurynVectorFloat(NeuronID n);

			/*! \brief Default destructor */
			~AurynVectorFloat() 
			{
			};

			void scale(float a);
			void saxpy(float a, AurynVectorFloat * x);
			void clip(float min, float max);
			void add(float c);
			void add(AurynVectorFloat * v);


			void mul(float a) { scale(a); };
			void mul(AurynVectorFloat * v);

	};

}


#endif /*AURYNVECTOR_H_*/
