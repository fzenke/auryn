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
	template <typename T, typename IndexType = NeuronID > 
	class AurynVector { 
		private: 
			friend class boost::serialization::access;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & size;
				for ( IndexType i = 0 ; i < size ; ++i ) 
					ar & data[i];
			}

		protected:
			/*! \brief Checks if argument is larger than size and throws and exception if so 
			 *
			 * Check only enabled if NDEBUG is not defined.*/
			void check_size(IndexType x)
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
			void check_size(AurynVector * v)
			{
#ifndef NDEBUG
				if ( v->size != size ) {
					throw AurynVectorDimensionalityException();
				}
#endif 
			}

			/*! \brief Implements aligned memory allocation */
			void allocate(const NeuronID n) {
				T * ptr = (T*)aligned_alloc(sizeof(T)*SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS,sizeof(T)*n);
				if ( ptr == NULL ) {
					// TODO implement proper exception handling
					throw AurynMemoryAlignmentException(); 
				}
				data = ptr;
				size = n;
				set_zero();
			}

			void freebuf() {
				free(data);
			}
		protected:

		public:
			IndexType size;
			T * data;

			/*! \brief Default constructor */
			AurynVector(IndexType n) 
			{
				allocate(n);
			}

			/*! \brief Copy constructor 
			 *
			 * Constructs vector as a copy of argument vector. */
			AurynVector(AurynVector * vec) 
			{
				allocate(vec->size);
				copy(vec);
			}


			/*! \brief Default destructor */
			virtual ~AurynVector() 
			{
				freebuf();
			}

			/*! \brief resize data array to new_size */
			void resize(IndexType new_size) 
			{
				if ( size != new_size ) {
					freebuf();
					allocate(new_size);
				}
				set_zero(); 
			}

			/*! \brief Set all elements to value v. */
			void set_all(T v) 
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] = v;
				}
			}

			/*! \brief Set all elements to zero. */
			void set_zero() 
			{
				set_all(0.0);
			}

			/*! \brief Scales all vector elements by a. */
			virtual void scale(const AurynFloat a) 
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] *= a;
				}
			}

			/*! \brief Scales all vector elements by a. TODO */
			void follow(AurynVector<T,IndexType> * v, const float rate) 
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] += rate*(v->data[i]-data[i]);
				}
			}

			/*! \brief Takes each element to the n-th power. 
			 *
			 * \param n the exponent */
			void pow(const unsigned int n) 
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] = std::pow(data[i],n);
				}
			}

			/*! \brief Takes the square root of each element  
			 *
			 * */
			void sqrt() 
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] = std::sqrt(data[i]);
				}
			}

			/*! \brief Adds constant c to each vector element */
			virtual void add(const AurynFloat c) 
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] += c;
				}
			}

			/*! \brief Adds the value c to specific vector element i */
			void add_specific(const IndexType i, const AurynFloat c) 
			{
				check_size(i);
				data[i] += c;
			}

			/*! \brief Adds a vector v to the vector
			 *
			 * No checking of the dimensions match! */
			void add(AurynVector * v) 
			{
				check_size(v);
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] += v->data[i];
				}
			}

			/*! \brief Subtract constant c to each vector element */
			void sub(const AurynFloat c) 
			{
				add(-c);
			}

			/*! \brief Elementwise subtraction */
			void sub(AurynVector * v) 
			{
				check_size(v);
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] -= v->data[i];
				}
			}

			/*! \brief Multiply all vector elements by constant */
			void mul(const AurynFloat a) 
			{
				scale(a);
			}

			/*! \brief Element-wise vector multiply  
			 *
			 * */
			void mul(AurynVector * v) 
			{
				check_size(v);
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] *= v->data[i];
				}
			}

			/*! \brief Computes the sum a+b and stores the result in this instance 
			 *
			 * */
			void sum(AurynVector * a, AurynVector * b) 
			{
				check_size(a);
				check_size(b);
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] = a->data[i]+b->data[i];
				}
			}

			/*! \brief Computes the sum a+b and stores the result in this instance 
			 *
			 * */
			void sum(AurynVector * a, const AurynState b) 
			{
				check_size(a);
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] = a->data[i]+b;
				}
			}

			/*! \brief Computes the difference a-b and stores the result in this instance 
			 *
			 * */
			void diff(AurynVector * a, AurynVector * b) 
			{
				check_size(a);
				check_size(b);
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] = a->data[i]-b->data[i];
				}
			}

			/*! \brief Computes the difference a-b and stores the result in this instance 
			 *
			 * */
			void diff(AurynVector * a, const AurynState b) 
			{
				sum(a,-b);
			}


			/*! \brief Copies vector v 
			 *
			 * */
			void copy(AurynVector * v) 
			{
				check_size(v);
				std::copy(v->data, v->data+v->size, data);
			}


			/*! \brief SAXPY operation as in GSL 
			 *
			 * Computes a*x + y and stores the result to y where y is the present instance. 
			 * \param a The scaling factor for the additional vector
			 * \param x The additional vector to add
			 * */
			virtual void saxpy(const AurynFloat a, AurynVector * x) 
			{
				check_size(x);
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] += a * x->data[i];
				}
			}

			/*! \brief Gets element i from vector */
			T get(IndexType i)
			{
				check_size(i);
				return data[i];
			}

			/*! \brief Gets pointer to element i from vector 
			 *
			 * When no argument is given the function returns the first element of 
			 * data array of the vector. */
			T * ptr(IndexType i = 0)
			{
				check_size(i);
				return data+i;
			}

			/*! \brief Sets element i in vector to value */
			void set(IndexType i, T value)
			{
				check_size(i);
				data[i] = value;
			}

			/*! \brief Squares each element 
			 *
			 * */
			void sqr() 
			{	
				this->mul(this);
			}

			/*! \brief Rectifies all elements
			 */
			void rect()
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					if ( data[i] < 0.0 ) {
						 data[i] = 0.0;
					} 
				}
			}

			/*! \brief Negatively rectifies all elements
			 */
			void neg_rect()
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					if ( data[i] > 0.0 ) {
						 data[i] = 0.0;
					} 
				}
			}

			/*! \brief Clips all vector elements to the range min max
			 *
			 * \param min Minimum value
			 * \param max Maximum value
			 */
			virtual void clip(T min, T max)
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					if ( data[i] < min ) {
						 data[i] = min;
					} else 
						if ( data[i] > max ) 
							 data[i] = max;
				}
			}

			/*! \brief Computes the variance of the vector elements 
			 *
			 * Uses Bessel's correction to calculate an unbiased estimate of the population variance which 
			 * requires n > 1 otherwise the output is not defined.
			 */
			double var()
			{
				double sum = 0.0;
				double sum2 = 0.0;
				for ( IndexType i = 0 ; i < size ; ++i ) {
					double elem = get(i);
					sum  += elem;
					sum2 += std::pow(elem,2);
				}
				double var  = (sum2-(sum*sum)/size)/(size-1);
				return var;
			}


			/*! \brief Computes the standard deviation of all elements
			 *
			 */
			double std()
			{
				return std::sqrt(var());
			}

			/*! \brief Computes the mean of the vector elements
			 *
			 */
			double mean()
			{
				double sum = 0.0;
				for ( IndexType i = 0 ; i < size ; ++i ) {
					sum += get(i);
				}
				return sum/size;
			}

			/*! \brief Computes number of nonzero elements
			 *
			 */
			IndexType nonzero()
			{
				IndexType sum = 0;
				for ( IndexType i = 0 ; i < size ; ++i ) {
					if ( get(i) != 0 ) ++sum;
				}
				return sum;
			}

			/*! \brief Print vector elements to std out for debugging */
			void print() {
				for ( IndexType i = 0 ; i < size ; ++i ) {
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
	class AurynVectorFloat : public AurynVector<float,NeuronID> 
	{
		private:

		public:
			/*! \brief Default constructor */
			AurynVectorFloat(NeuronID n);

			/*! \brief Default destructor */
			~AurynVectorFloat() 
			{
			};


			void scale(const float a);
			void saxpy(const float a, AurynVectorFloat * x);
			void clip(const float min, const float max);
			void add(const float c);
			void add(AurynVectorFloat * v);
			void sum(AurynVectorFloat * a, AurynVectorFloat * b);
			void sum(AurynVectorFloat * a, const AurynState b);
			void diff(AurynVectorFloat * a, AurynVectorFloat * b);
			void diff(AurynVectorFloat * a, const AurynState b);


			// TODO add pow function with intrinsics _mm_pow_ps


			void mul(const float a) { scale(a); };
			void mul(AurynVectorFloat * v);

	};

}


#endif /*AURYNVECTOR_H_*/
