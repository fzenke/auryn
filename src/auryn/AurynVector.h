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

#include <ctime>
#include <assert.h>
#include "auryn_definitions.h"


namespace auryn {

	/*! \brief Default Auryn vector template 
	 *
	 * This class implements the base vector class for Auryn.
	 * Neuronal or synaptic state variables are stored as AurynVectors.
	 * For performance reasons, time critical functions of this template have to be reimplemented in derived 
	 * classes with a specific template parameter T. For instance I will always provide a derived type AurynVectorFloat (T=float)
	 * which per default is synonymous to AurynStateVector which implements SSE instructions for labour 
	 * intensive operations on the vectors. When using the AurynStateVector in NeuronGroups etc this will speed up computation 
	 * performance automatically.
	 * Note, that all Auryn vectors should initialized with multiple of 4 elements (later that number might change) when
	 * we add AVX support to the code. This is done automatically \if you rely on the get_state_vector function implemented in SpikingGroup.
	 * Alternatively, if you use get_vector_size functions from SpikingGroup this will automatically
	 * be taken care of too.
	 * */
	template <typename T, typename IndexType = NeuronID > 
	class AurynVector { 
		private: 

			/*!\brief Pointer to allocated unaligned memory */
			void * mem;

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
#ifdef CODE_ALIGNED_SIMD_INSTRUCTIONS
				std::size_t mem_alignment = sizeof(T)*SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS;
				std::size_t mem_size = sizeof(T)*n;
				mem = malloc(mem_size+mem_alignment-1); // adds padding to allocated memory
				T * ptr = (T*)mem; 
				if ( (unsigned long)mem%mem_alignment ) ptr = (T*)(((unsigned long)mem/mem_alignment+1)*mem_alignment);
				//! \todo TODO Replace above alignment code with boost code once boost 1.56 is commonly available with the dists
				// T * ptr = (T*)boost::alignment::aligned_alloc(sizeof(T)*SIMD_NUM_OF_PARALLEL_FLOAT_OPERATIONS,sizeof(T)*n);
				if ( mem == NULL ) {
					// TODO implement proper exception handling
					throw AurynMemoryAlignmentException(); 
				}
				assert(((unsigned long)ptr % mem_alignment) == 0);
#else
				T * ptr = new T[n];
#endif
				data = ptr;
				size = n;
				set_zero();
			}

			void freebuf() {
#ifdef CODE_ALIGNED_SIMD_INSTRUCTIONS
				free(mem);
				//! \todo TODO Replace above alignment code with boost code once boost 1.56 is commonly available with the dists
				// boost::alignment::aligned_free(data);
#else
				delete [] data;
#endif
			}

            /*! \brief Computes approximation of exp(x) via fast series approximation up to n=256. */
            T fast_exp256(T x)
            {
                    x = 1.0 + x / 256.0;
                    x *= x; x *= x; x *= x; x *= x;
                    x *= x; x *= x; x *= x; x *= x;
                    return x;
            }

		protected:

		public:
			/*! \brief Size of the vector
			 *
			 * \todo Consider including a non_zero size paramter too,
			 * because we are using this template also in sparse matrices now
			 * for complex synaptic dynamics in which not all elements are necessarily
			 * used...
			 * */
			IndexType size;

			/*! \brief Pointer to the array housing the data */
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

			/*! \brief resize data array to new_size 
			 *
			 * The function tries to preserve data while resizing. 
			 * If a vector is downsized elements at the end are simply dropped.
			 * When the vector size is increased the new elements at the end are
			 * intialized with zeros.*/
			virtual void resize(IndexType new_size) 
			{
				if ( size != new_size ) {
					T * old_data = data;
					IndexType old_size = size;
					allocate(new_size);
					// copy old data
					const size_t copy_size = std::min(old_size,new_size) * sizeof(T);
					std::memcpy(data, old_data, copy_size);
					free(old_data);
				}
			}

			/*! \brief Copies vector v 
			 *
			 * */
			void copy(AurynVector * v) 
			{
				check_size(v);
				std::copy(v->data, v->data+v->size, data);
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


			/*! \brief Set all elements to value v. */
			void set_all(const T v) 
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
			void scale(const T a) 
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] *= a;
				}
			}


			/*! \brief Adds constant c to each vector element */
			void add(const T c) 
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] += c;
				}
			}

			/*! \brief Adds the value c to specific vector element i */
			void add_specific(const IndexType i, const T c) 
			{
				check_size(i);
				data[i] += c;
			}

			/*! \brief Multiply to specific vector element with data index i with the constant c*/
			void mul_specific(const IndexType i, const T c) 
			{
				check_size(i);
				data[i] *= c;
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
			void sub(const T c) 
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
			void mul(const T a) 
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

			/*! \brief SAXPY operation as in GSL 
			 *
			 * Computes a*x + y and stores the result to y where y is the present instance. 
			 * \param a The scaling factor for the additional vector
			 * \param x The additional vector to add
			 * */
			void saxpy(const T a, AurynVector * x) 
			{
				check_size(x);
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] += a * x->data[i];
				}
			}


			/*! \brief Scales all vector elements by a. */
			void follow(AurynVector<T,IndexType> * v, const T rate) 
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

			/*! \brief Computes an approximation of exp(x) for each vector element. 
			 *
			 * */
			void fast_exp()
			{
				// mul(0.00390625); // ie. 1.0/256.0
				// add(1.0);
				// for ( int i = 0 ; i < 8 ; ++i ) {
				// 	sqr();
				// }
				// seems as if the naive version is faster
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] = fast_exp256(data[i]);
				}
			}

			/*! \brief Computes exp(x) for each vector element. */
			void exp()
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] = std::exp(data[i]);
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

			/*! \brief Flips the sign of all elements. */
			void neg()
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					data[i] = -data[i];
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
			void sum(AurynVector * a, const T b) 
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
			void diff(AurynVector * a, const T b) 
			{
				sum(a,-b);
			}

			/*! \brief Computes the difference a-b and stores the result in this instance 
			 *
			 * */
			void diff(const T a, AurynVector * b) 
			{
				check_size(b);
				sum(b,-a);
				neg();
			}


			/*! \brief Squares each element 
			 *
			 * */
			void sqr() 
			{	
				this->mul(this);
			}

			/*! \brief Takes absolute value of each element
			 *
			 * */
			void abs() 
			{	
				sqr();
				sqrt();
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
			void clip(T min, T max)
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					if ( data[i] < min ) {
						 data[i] = min;
					} else 
						if ( data[i] > max ) 
							 data[i] = max;
				}
			}

			/*! \brief Computes the variance of the vector elements on this rank
			 *
			 * Uses Bessel's correction to calculate an unbiased estimate of the population variance which 
			 * requires n > 1 otherwise the output is not defined.
			 *
			 * Warning: Note that AurynVector can only compute the mean of all the subset of
			 * elements stored on rank it runs on.
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


			/*! \brief Computes the standard deviation of all elements on this rank
			 *
			 * Warning: Note that AurynVector can only compute the mean of all the subset of
			 * elements stored on rank it runs on.
			 */
			double std()
			{
				return std::sqrt(var());
			}

			/*! \brief Computes the mean of the vector elements on this rank
			 *
			 * Warning: Note that AurynVector can only compute the mean of all the subset of
			 * elements stored on rank it runs on.
			 */
			double mean()
			{
				return element_sum()/size;
			}

			/*! \brief Computes the sum of the vector elements
			 *
			 */
			double element_sum()
			{
				double sum = 0.0;
				for ( IndexType i = 0 ; i < size ; ++i ) {
					sum += get(i);
				}
				return sum;
			}

			/*! \brief Computes number of nonzero elements on this rank
			 *
			 * Warning: Note that AurynVector can only compute the mean of all the subset of
			 * elements stored on rank it runs on.
			 */
			IndexType nonzero()
			{
				IndexType sum = 0;
				for ( IndexType i = 0 ; i < size ; ++i ) {
					if ( get(i) != 0 ) ++sum;
				}
				return sum;
			}

			/*! \brief Sets all values whose absolute value is smaller than epsilon to zero.
			 *
			 */
			void zero_effective_zeros( const T epsilon = 1e-3)
			{
				for ( IndexType i = 0 ; i < size ; ++i ) {
					if ( std::abs(get(i)) < epsilon ) set(i, 0.0);
				}
			}

			void set_random_normal(AurynState mean=0.0, AurynState sigma=1.0, unsigned int seed=8721)
			{
				if ( seed == 0 )
					seed = static_cast<unsigned int>(std::time(0));
				boost::mt19937 randgen(seed); 
				boost::normal_distribution<> dist((double)mean, (double)sigma);
				boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(randgen, dist);
				AurynState rv;
				for ( IndexType i = 0 ; i<size ; ++i ) {
					rv = die();
					data[i] = rv;
				}
			}

			/*! \brief Initializes vector elements with Gaussian of unit 
			 * varince and a seed derived from system time if no seed or seed of 0 is given. */
			void set_random(unsigned int seed = 0) 
			{
				set_random_normal(0.0,1.0,seed);
			}


			/*! \brief Print vector elements jo std out for debugging */
			void print() {
				for ( IndexType i = 0 ; i < size ; ++i ) {
					std::cout << get(i) << " ";
				}
				std::cout << std::endl;
			}
	};




	/*! \brief Default AurynVectorFloat class for performance computation
	 *
	 * This class derives from AurynVector<float,NeuronID> and overwrites 
	 * some performance critical member functions defined in the template 
	 * with SIMD intrinsics for higher performance.
	 */
	class AurynVectorFloat : public AurynVector<float,NeuronID> 
	{
		private:
			typedef AurynVector<float,NeuronID> super;

		public:
			/*! \brief Default constructor */
			AurynVectorFloat(NeuronID n);

			/*! \brief Default destructor */
			~AurynVectorFloat() 
			{
			};


			virtual void resize(NeuronID new_size);
			void scale(const float a);
			void saxpy(const float a, AurynVectorFloat * x);
			void clip(const float min, const float max);
			void add(const float c);
			void add(AurynVectorFloat * v);
			void sum(AurynVectorFloat * a, AurynVectorFloat * b);
			void sum(AurynVectorFloat * a, const float b);
			void mul(const float a) { scale(a); };
			void mul(AurynVectorFloat * v);
			void diff(AurynVectorFloat * a, AurynVectorFloat * b);
			void diff(AurynVectorFloat * a, const float b);
			void diff(const float a, AurynVectorFloat * b );
			void follow(AurynVectorFloat * v, const float rate);

			// TODO add pow function with intrinsics _mm_pow_ps

	};

}


#endif /*AURYNVECTOR_H_*/
