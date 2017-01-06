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

#ifndef COMPLEXMATRIX_H_
#define COMPLEXMATRIX_H_

#include "auryn_global.h"
#include "auryn_definitions.h"

#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

namespace auryn {

/*! \brief Template for a sparse matrix with row major ordering and fast access
 * of rows and capability to store float values per matrix entry.
 *
 * This matrix class implements a sparse matrix that allows fast access of all
 * elements of one row. It provides row interators to efficiently propagate
 * spikes.  Memory has to be reserved when the class is defined and elements can
 * only be inserted row by row starting from "left to right". This scheme
 * enables related data fields to reside in memory next to each other.
 *
 * ComplexMatrix generalizes SimpleMatrix to a rank 3 tensor in which each synaptic 
 * connection can have more than one value (third tensor mode -- corresponding 
 * to a synaptic state). This allows to efficiently implement state based complex 
 * synaptic models which have their own internal dynamics. 
 *
 * Instead of storing all synaptic data values in one long array the synaptic state 
 * values are stored in multiple state vectors which can be manipulated efficiently
 * in a vector based manner.
 *
 * For instance, suppose w holds your instance of ComplexMatris in which you have 
 * set_num_synapse_states(3) which makes it a connectivity matrix which reserves three
 * state variables of type AurynWeight per synaptic connection. Now, decaying all elements 
 * of, say, state 3 by a factor "foo" is as simple as w->get_state_vector(2)->scale(foo).
 * Adding two of the states for all synapses becomes 
 * w->get_state_vector(2)->add(w->get_state_vector(3)).
 *
 * Because these expressions become quickly rather lenghty it is nice to declare shortcuts
 * in your plastic Connection class such as:
 * AurynSynStateVector * w_val = w->get_state_vector(0);
 * AurynSynStateVector * tagging_val = w->get_state_vector(1);
 * AurynSynStateVector * scaffold_val = w->get_state_vector(2);
 *
 * You can now work with these as you are used with AurynVector or AurynStateVector instances 
 * w_val->saxpy(foo,tagging_val)
 *
 */


template <typename T>
class ComplexMatrix 
{
private:

	friend class boost::serialization::access;
	template<class Archive>
	void save(Archive & ar, const unsigned int version) const
		{
			ar & m_rows;
			ar & n_cols;
			ar & n_z_values;
			ar & current_row;
			ar & current_col;
			ar & statesize;
			ar & n_nonzero;

			// rowpointers -- translate in elements per row
			for ( NeuronID i = 0 ; i < m_rows+1 ; ++i ) {
				NeuronID num_elements = rowptrs[i]-rowptrs[0];
				ar & num_elements;
			}
			// colindices
			for (AurynLong i = 0 ; i < n_nonzero ; ++i) {
				ar & colinds[i];
			}
			// data
			for (StateID z = 0; z < n_z_values ; ++z ) {
				ar & *(statevectors[z]);
			}
		}
	template<class Archive>
	void load(Archive & ar, const unsigned int version)
		{
			const StateID cur_num_syn_states = get_num_synaptic_states();

			ar & m_rows;
			ar & n_cols;
			ar & n_z_values;
			ar & current_row;
			ar & current_col;
			ar & statesize;
			ar & n_nonzero;

			if ( n_z_values != cur_num_syn_states ) { // check if we have the some number of tensor modes
				throw AurynMatrixComplexStateException();
			}

			// allocate necessary memory
			resize_buffers(statesize);

			// rowpointers -- translate in elements per row
			for ( NeuronID i = 0 ; i < m_rows+1 ; ++i ) {
				NeuronID num_elements;
				ar & num_elements;
				rowptrs[i] = rowptrs[0]+num_elements;
			}
			// colindices
			for (AurynLong i = 0 ; i < n_nonzero ; ++i) {
				ar & colinds[i];
			}
			// data
			for (StateID z = 0; z < n_z_values ; ++z ) {
				ar & *(statevectors[z]);
			}
		}
	BOOST_SERIALIZATION_SPLIT_MEMBER()


	AurynLong data_index_error_value;

	/*! \brief Dimensions of matrix */
	NeuronID m_rows,n_cols;
	/*! \brief z-dimension of matrix */
	NeuronID n_z_values;
	/*! \brief Current row and col for filling. Should be in the last corner before use. */
	NeuronID current_row, current_col;
	/*! \brief The number of actual nonzero elements. */
	AurynLong n_nonzero;
	/*! \brief The size of data reserved and therefore the maximum number of non-zero elements. */
	AurynLong statesize;

protected:
	/*! \brief Array that holds the begin addresses of column indices */
	NeuronID ** rowptrs;

	/*! \brief Array that holds the column indices of non-zero elements */
	NeuronID * colinds;

	/*! \brief Returns a synaptic state vector. */
	AurynVector<T,AurynLong> * alloc_synaptic_state_vector();

	/*! \brief Matches size of statevectors to number of synaptic states.  */
	void prepare_state_vectors();

	/*! \brief Default initializiation called by the constructor. */
	void init(NeuronID m, NeuronID n, AurynLong size, NeuronID z );
	void free();

public:
	/*! \brief Vector that holds pointers to the state vectors storing the synaptic states. */
	std::vector< AurynVector<T,AurynLong> * > statevectors;

	/*! \brief Empty constructor */
	ComplexMatrix();

	/*! \brief Copy constructor */
	ComplexMatrix(ComplexMatrix * mat);

	/*! \brief Default constructor */
	ComplexMatrix(NeuronID rows, NeuronID cols, AurynLong size=256, StateID n_values=1 );

	/*! \brief Default destructor */
	virtual ~ComplexMatrix();

	/*! \brief Clears matrix */
	void clear();

	/*! \brief Returns pointer to statevector which is an AurynVector of specified synaptic state */
	AurynVector<T,AurynLong> * get_synaptic_state_vector(StateID z=0);

	/*! \brief Sames as get_synaptic_state_vector(StateID z) */
	AurynVector<T,AurynLong> * get_state_vector(StateID z=0);

	/*! \brief Resize buffer
	 *
	 *  Allocates a new buffer of size and copies the 
	 *  old buffers before freeing the memory */
	void resize_buffers(AurynLong size);

	/*! \brief Resizes buffer and clears the matrix. This saves
	 *  to copy all the data. */
	void resize_buffer_and_clear(AurynLong size);

	/*! \brief Prunes the reserved memory such that datasize=get_nonzero()
	 *  Note that this is an expensive operation because it uses
	 *  resize_buffers(size) and it should be used sparsely. */
	void prune();

	/*! Add non-zero element in row/col order. 
	 * \param i row index where to insert element 
	 * \param j col index where to insert element
	 * \param value to insert
	 * \throw AurynMatrixBufferException */
	void push_back(const NeuronID i, const NeuronID j, const T value);

	/*! \brief Copies complex matrix mat */
	void copy(ComplexMatrix * mat);


	/* \brief Sets element identified by the data index to value */
	void set_element(AurynLong data_index, T value, StateID z=0);

	/* Deprecated: TODO Remove this function */
	void set_data(AurynLong i, T value, StateID z=0);


	/* \brief Scales element identified by the data index by value */
	void scale_element(AurynLong data_index, T value, StateID z=0);

	/* Deprecated: TODO Remove this function */
	void scale_data(AurynLong i, T value, StateID z=0);

	/*! Gets the matching data entry for a given index i and state z*/
	T get_data(AurynLong i, StateID z=0);

	/*! \brief Gets the matching data ptr for a given index i and state z*/
	T * get_data_ptr(const AurynLong data_index, const StateID z=0);
	/*! \brief Gets the matching data ptr for a given index pointer and state z*/
	T * get_data_ptr(const NeuronID * ind_ptr, const StateID z=0);

	/*! \brief Gets the matching data value for a given index pointer and state z*/
	T get_data(const NeuronID * ind_ptr, StateID z=0);

	/*! \brief Pads non-existing elements for the remaining elements to a matrix
	 *
	 * This function has to be called after filling the matrix with elements e.g. random sparse
	 * and before using it in a simulation. It is typically called by the finalize function in 
	 * SparseConnection. */
	void fill_zeros();


	/*! \brief Returns the fill level of the matrix element buffer 
	 *
	 * A fill level of 1.0 corresponds to the sparse matrix element buffer
	 * being full.*/
	AurynDouble get_fill_level();

	/*! \brief Value of synaptic state variable i,j,z returns zero if the element is zero or does not exist. */
	T get(const NeuronID i, const NeuronID j, const NeuronID z=0);

	/*! \brief Returns true if the matrix element exists. */
	bool exists(const NeuronID i, const NeuronID j, const StateID z=0);

	/*! \brief Returns the pointer to a particular element */
	T * get_ptr(const NeuronID i, const NeuronID j, const StateID z=0);

	/*! \brief Same as get_data_ptr. Returns the pointer to a particular element given 
	 * its position in the data array. 
	 *
	 * \todo TODO For two args this function is ambiguous with the above get_ptr def -- need to find a better way for that
	 * */
	T * get_ptr(const AurynLong data_index, const StateID z=0);

	/*! \brief Returns a particular element given 
	 * its position in the data array. */
	T get_element(const AurynLong data_index, const StateID z);

	/*! \brief Returns data index to a particular element specifed by an index pointer */
	AurynLong ind_ptr_to_didx(const NeuronID * ind_ptr);
	/*! \brief Returns data index to a particular element specifed by an index pointer */
	AurynLong get_data_index(const NeuronID * ind_ptr);
	/*! Returns data index to a particular element specifed by i and j */
	AurynLong get_data_index(const NeuronID i, const NeuronID j);
	/*! \brief Returns data index to a particular element specifed by a data pointer */
	AurynLong data_ptr_to_didx(const T * ptr);
	/*! \brief Returns data index to a particular element specifed by a data pointer */
	AurynLong get_data_index(const T * ptr) { return data_ptr_to_didx(ptr); };


	/* Methods concerning synaptic state vectors. */

	/*!\brief Sets number of synaptic states (z-value) */
	void set_num_synaptic_states(const StateID zsize);
	/*!\brief Sets number of synaptic states (z-value) */
	void set_num_synapse_states(const StateID zsize);
	/*!\brief Returns number of synaptic states (z-value) */
	StateID get_num_synaptic_states();
	/*!\brief Returns number of synaptic states (z-value) */
	StateID get_num_z_values();
	/*!\brief Synonymous to get_num_synaptic_states */
	StateID get_num_synapse_states();

	/*! \brief Gets pointer for the first element of a given synaptic state vector */
	T * get_state_begin(const StateID z=0);
	/*! \brief Gets pointer for the element behind the last of a given synaptic state vector */
	T * get_state_end(const StateID z=0);
	/*! \brief Sets all values in state x to value. */
	void state_set_all(T * x, const T value);
	/*! \brief Computes a*x + y and stores result in y */
	void state_saxpy(const T a, T * x, T * y);
	/*! \brief Multiplies x and y and stores result in y */
	void state_mul(T * x, T * y);
	/*! \brief Adds x and y and stores result in y */
	void state_add(T * x, T * y);
	/*! \brief Computes x-y and stores result in y */
	void state_sub(T * x, T * y);
	/*! \brief Computes x-y and stores result in res */
	void state_sub(T * x, T * y, T * res);
	/*! \brief Scale state x by a. */
	void state_scale(const T a, T * x);
	/*! \brief Adds constant a to all values in x */
	void state_add_const(const T a, T * x);
	/*! \brief Clips state values to interval [a,b] */
	void state_clip(T * x, const T a, const T b);
	/*! \brief Get data pointer for that state */
	T * state_get_data_ptr(T * x, NeuronID i);


	void add_value(const AurynLong data_index, T value);
	NeuronID get_colind(const AurynLong data_index);

	bool set(const NeuronID i, const NeuronID j, T value);
	/*! \brief Sets all non-zero elements to value */
	void set_all(const T value, const StateID z=0);
	/*! \brief Sets all non-zero elements in row i to value */
	void set_row(const NeuronID i, const T value);
	/*! \brief Scales all non-zero elements in row i to value */
	void scale_row(const NeuronID i, const T value);
	/*! \brief Scales all non-zero elements */
	void scale_all(const T value);
	/*! \brief Sets all non-zero elements in col j to value. Due to ordering this is slow and the use of this functions is discouraged. */
	void set_col(const NeuronID j, const T value);
	/*! \brief Scales all non-zero elements in col j to value. Due to ordering this is slow and the use of this functions is discouraged. */
	void scale_col(const NeuronID j, const T value);
	double sum_col(const NeuronID j);
	/*! \brief Returns datasize: number of possible entries */
	AurynLong get_datasize();
	/*! \brief Same as datasize : number of possible entries */
	AurynLong get_statesize();
	/*! \brief Returns statesize multiplied by number of states */
	AurynLong get_memsize();
	/*! \brief Returns number of non-zero elements */
	AurynLong get_nonzero();
	/*! \brief stdout dump of all elements -- for testing only. */
	void print();

	/*! \brief Return mean value of elements for the first complex state (z=0). 
	 *
	 * Warning: Note that ComplexMatrix can only compute the mean of all the subset of
	 * elements stored on thank it runs on. */
	double mean();

	NeuronID * get_ind_begin();
	NeuronID * get_row_begin(NeuronID i);
	AurynLong get_row_begin_index(NeuronID i);
	NeuronID * get_row_end(NeuronID i);
	AurynLong get_row_end_index(NeuronID i);
	NeuronID get_m_rows();
	NeuronID get_n_cols();
	NeuronID ** get_rowptrs();
	/*! \brief Returns pointer to data value corresponding to the first element. */
	T * get_data_begin(const StateID z=0);
	/*! \brief Returns pointer to data value corresponding to the element behind the last nonzero element. */
	T * get_data_end(const StateID z=0);
	/*! \brief Returns the data value to an item that is i-th in the colindex array */
	T get_value(const AurynLong data_index);
	/*! \brief Returns the data value to an item that for pointer r pointing to the respective element in the index array */
	T get_value(NeuronID * r);
	/*! \brief Returns pointer to the the data value to an item that is i-th in the colindex array */
	T * get_value_ptr(const NeuronID i);
	/*! \brief Returns the pointer to the data value to an item that for pointer r pointing to the respective element in the index array */
	T * get_value_ptr(NeuronID * r);
	NeuronID get_data_offset(NeuronID * r);
};

template <typename T>
T * ComplexMatrix<T>::get_data_ptr(const AurynLong data_index, const StateID z)
{
	return statevectors[z]->data+data_index;
}

template <typename T>
T ComplexMatrix<T>::get_data(const AurynLong i, const StateID z)
{
	return *get_data_ptr(i,z);
}

template <typename T>
T * ComplexMatrix<T>::get_data_ptr(const NeuronID * ind_ptr, const StateID z) 
{
	size_t ptr_offset = ind_ptr-get_ind_begin();
	return statevectors[z]->data+ptr_offset;
}

template <typename T>
T ComplexMatrix<T>::get_data(const NeuronID * ind_ptr, StateID z) 
{
	return *(get_data_ptr(ind_ptr,z));
}

template <typename T>
T ComplexMatrix<T>::get_element(const AurynLong data_index, const StateID z)
{
	return *get_data_ptr(data_index);
}

template <typename T>
void ComplexMatrix<T>::set_element(AurynLong data_index, T value, StateID z)
{
	if (data_index<statesize)
		statevectors[z]->data[data_index] = value;
}


template <typename T>
void ComplexMatrix<T>::set_data(AurynLong i, T value, StateID z)
{
	set_element(i,value,z);
}

template <typename T>
void ComplexMatrix<T>::scale_element(AurynLong data_index, T value, StateID z)
{
	if (data_index<statesize)
		statevectors[z]->data[data_index] *= value;
}

template <typename T>
void ComplexMatrix<T>::scale_data(AurynLong i, T value, StateID z)
{
	scale_element(i, value, z);
}

template <typename T>
void ComplexMatrix<T>::clear()
{
	current_row = 0;
	current_col = 0;
	n_nonzero = 0;
	rowptrs[0] = colinds;
	rowptrs[1] = colinds;
}

template <typename T>
AurynVector<T,AurynLong> * ComplexMatrix<T>::alloc_synaptic_state_vector()
{
	T * vec = new AurynVector<T,AurynLong>(get_statesize());
	return vec;
}

template <typename T>
AurynVector<T,AurynLong> * ComplexMatrix<T>::get_synaptic_state_vector(StateID z)
{
	if ( z >= get_num_synaptic_states() ) {
		logger->error("Trying to access a complex state larger than number of states in the tensor");
		throw AurynMatrixDimensionalityException();
	}
	return statevectors[z];
}

template <typename T>
void ComplexMatrix<T>::prepare_state_vectors()
{
	// add synaptic state vectors until we have enough
	while ( statevectors.size() < get_num_synaptic_states() ) {
		AurynVector<T,AurynLong> * vec = new AurynVector<T,AurynLong>(get_statesize());
		statevectors.push_back(vec);
	}

	// remove state vectors from the end until we have the right amount
	// if the above code ran this will not run
	while ( statevectors.size() > get_num_synaptic_states() ) {
		delete [] *(statevectors.end());
		statevectors.pop_back();
	}
}

template <typename T>
void ComplexMatrix<T>::init(NeuronID m, NeuronID n, AurynLong size, NeuronID z)
{
	m_rows = m;
	n_cols = n;

	statesize = size; // assumed maximum number of connections
	n_z_values = z;

	data_index_error_value = std::numeric_limits<AurynTime>::max();
	

	rowptrs = new NeuronID * [m_rows+1];
	colinds = new NeuronID [get_datasize()];
	prepare_state_vectors(); 
	clear();
}

template <typename T>
void ComplexMatrix<T>::resize_buffers(AurynLong new_size)
{
	AurynLong oldsize = get_statesize();
	if ( oldsize == new_size ) return;
	statesize = new_size;

	NeuronID * new_colinds = new NeuronID [get_datasize()];
	std::copy(colinds, colinds+get_nonzero(), new_colinds);

	// update rowpointers
	ptrdiff_t offset = new_colinds-colinds;
	for ( NeuronID i = 0 ; i < m_rows+1 ; ++i ) {
		rowptrs[i] += offset;
	}
	
	delete [] colinds;
	colinds = new_colinds;

	for ( StateID i = 0 ; i < get_num_synaptic_states() ; ++i ) {
		statevectors[i]->resize(new_size);
	}
}

template <typename T>
void ComplexMatrix<T>::resize_buffer_and_clear(AurynLong size)
{
	free();
	init(m_rows, n_cols, size, n_z_values );
}

template <typename T>
void ComplexMatrix<T>::prune()
{
	if ( get_datasize() > get_nonzero() )
		resize_buffers(get_nonzero());
}

template <typename T>
ComplexMatrix<T>::ComplexMatrix()
{
	init(1,1,2,1);
}

template <typename T>
ComplexMatrix<T>::ComplexMatrix(ComplexMatrix * mat)
{
	init(mat->get_m_rows(), mat->get_n_cols(), mat->get_nonzero(), mat->get_num_z_values() );
	copy(mat);
}

template <typename T>
ComplexMatrix<T>::ComplexMatrix(NeuronID rows, NeuronID cols, AurynLong statesize, StateID n_values )
{
	init(rows, cols, statesize, n_values);
}

template <typename T>
void ComplexMatrix<T>::free()
{
	delete [] rowptrs;
	delete [] colinds;
	while ( statevectors.size() ) {
		AurynVector<T,AurynLong> * vec = statevectors.back();
		delete vec;
		statevectors.pop_back();
	}
}

template <typename T>
void ComplexMatrix<T>::copy(ComplexMatrix * mat)
{
	if ( get_m_rows() != mat->get_m_rows() || get_n_cols() != mat->get_n_cols() )
		throw AurynMatrixDimensionalityException();

	clear();

	// copy sparse strcture and first state
	for ( NeuronID i = 0 ; i < mat->get_m_rows() ; ++i ) {
		for ( NeuronID * r = mat->get_row_begin(i) ; r != mat->get_row_end(i) ; ++r ) {
			push_back(i,*r,mat->get_value(r));
		}
	}
	
	// copy the other states
	for ( StateID z = 1 ; z < get_num_synaptic_states() ; ++z ) {
		statevectors[z]->copy(mat->statevectors[z]);
	}
}

template <typename T>
ComplexMatrix<T>::~ComplexMatrix()
{
	free();
}

template <typename T>
void ComplexMatrix<T>::push_back(NeuronID i, NeuronID j, T value)
{   
	if ( i >= m_rows || j >= n_cols ) throw AurynMatrixDimensionalityException();
	while ( i > current_row )
	{
		rowptrs[current_row+2] = rowptrs[current_row+1];  // save value of last element
		current_col = 0;
		current_row++;
	} 
	current_col = j;
	T * data_z0 = get_synaptic_state_vector(0)->data;
	if (i >= current_row && j >= current_col) {
		if ( n_nonzero >= get_datasize() ) throw AurynMatrixBufferException();
		*(rowptrs[i+1]) = j; // write last j to end of index array
		data_z0[rowptrs[i+1]-colinds] = value; // write value to end of data array
		++rowptrs[i+1]; //increment end by one
		rowptrs[m_rows] = rowptrs[i+1]; // last (m_row+1) marks end of last row
		n_nonzero++;
	} else {
		throw AurynMatrixPushBackException();
	}
}

template <typename T>
AurynLong ComplexMatrix<T>::get_datasize()
{
	return get_statesize();
}

template <typename T>
AurynLong ComplexMatrix<T>::get_statesize()
{
	return statesize;
}


template <typename T>
AurynLong ComplexMatrix<T>::get_memsize()
{
	return statesize*n_z_values;
}

template <typename T>
AurynLong ComplexMatrix<T>::get_nonzero()
{
	return n_nonzero;
}

template <typename T>
void ComplexMatrix<T>::fill_zeros()
{
	for ( NeuronID i = current_row ; i < m_rows-1 ; ++i )
	{
		rowptrs[i+2] = rowptrs[i+1];  // save value of last element
	}
	current_row = get_m_rows();
}


template <typename T>
bool ComplexMatrix<T>::exists(NeuronID i, NeuronID j, NeuronID z)
{
	if ( get_data_index(i,j) == data_index_error_value || z >= get_num_synaptic_states() )
		return false;
	else 
		return true;
}

template <typename T>
void ComplexMatrix<T>::set_num_synaptic_states(StateID zsize)
{
	n_z_values = zsize;
	prepare_state_vectors();
	resize_buffers(statesize);
}

template <typename T>
void ComplexMatrix<T>::set_num_synapse_states(StateID zsize)
{
	set_num_synaptic_states(zsize);
}

template <typename T>
StateID ComplexMatrix<T>::get_num_synaptic_states()
{
	return n_z_values;
}

template <typename T>
NeuronID ComplexMatrix<T>::get_num_z_values()
{
	return get_num_synaptic_states();
}

template <typename T>
NeuronID ComplexMatrix<T>::get_num_synapse_states()
{
	return get_num_synaptic_states();
}

template <typename T>
AurynVector<T,AurynLong> * ComplexMatrix<T>::get_state_vector(StateID z)
{
	return get_synaptic_state_vector(z);
}

template <typename T>
T * ComplexMatrix<T>::get_state_begin(StateID z)
{
	return statevectors[z]->data;
}


template <typename T>
T * ComplexMatrix<T>::get_state_end(StateID z)
{
	return get_state_begin(z)+get_datasize();
}


template <typename T>
AurynLong ComplexMatrix<T>::get_data_index(NeuronID i, NeuronID j)
{
#ifdef DEBUG
		std::cout << "cm: starting bisect " << i << ":" << j << std::endl;
#endif // DEBUG

	// check bounds
	if ( !(i < m_rows && j < n_cols) ) return data_index_error_value;

	// perform binary search
	NeuronID * lo = rowptrs[i];
	NeuronID * hi = rowptrs[i+1];
	NeuronID * c  = hi;

	if ( lo >= hi ) // no elements/targets in this row
		return data_index_error_value; 
	
	while ( lo < hi ) {
		c = lo + (hi-lo)/2;
		if ( *c < j ) lo = c+1;
		else hi = c;
#ifdef DEBUG
		std::cout << "cm: " << i << ":" << j << "   " << *lo << ":" << *hi << std::endl;
#endif // DEBUG
	}
	
	if ( *lo == j ) {
#ifdef DEBUG
		std::cout << "cm: " << "found element at data array position " 
			<< (lo-colinds) << std::endl;
#endif // DEBUG
		return (lo-colinds);
	}

#ifdef DEBUG
		std::cout << "cm: " << "element not found" << std::endl;
#endif // DEBUG

	return data_index_error_value; 
}



template <typename T>
AurynLong ComplexMatrix<T>::ind_ptr_to_didx(const NeuronID * ind_ptr)
{
	return ind_ptr - get_ind_begin();
}

template <typename T>
AurynLong ComplexMatrix<T>::get_data_index(const NeuronID * ind_ptr)
{
	return ind_ptr_to_didx(ind_ptr);
}

template <typename T>
AurynLong ComplexMatrix<T>::data_ptr_to_didx(const T * ptr)
{
	return ptr - get_data_begin();
}

template <typename T>
T * ComplexMatrix<T>::get_ptr(NeuronID i, NeuronID j, StateID z)
{
	AurynLong data_index = get_data_index(i,j);
	if ( data_index != data_index_error_value )
		return statevectors[z]->data+data_index;
	else
		return NULL;
}

template <typename T>
T ComplexMatrix<T>::get(NeuronID i, NeuronID j, StateID z)
{
	AurynLong data_index = get_data_index(i,j);
	if ( data_index != data_index_error_value )
		return *(statevectors[z]->data+get_data_index(i,j));
	else
		return 0;
}

template <typename T> 
T * ComplexMatrix<T>::get_ptr(const AurynLong data_index, const StateID z)
{
	return get_data_ptr(data_index, z);
}


template <typename T>
T ComplexMatrix<T>::get_value(AurynLong data_index)
{
	return *get_ptr(data_index);
}

template <typename T>
void ComplexMatrix<T>::add_value(AurynLong data_index, T value)
{
	*get_ptr(data_index) += value;
}

template <typename T>
NeuronID ComplexMatrix<T>::get_colind(AurynLong data_index)
{
	return colinds[data_index];
}

template <typename T>
bool ComplexMatrix<T>::set(NeuronID i, NeuronID j, T value)
{
	T * ptr = get_ptr(i,j);
	if ( ptr != NULL) {
		*ptr = value;
		return true;
	}
	else
		return false;
}

template <typename T>
void ComplexMatrix<T>::scale_row(NeuronID i, T value)
{
	NeuronID * rowbegin = rowptrs[i];
	NeuronID * rowend = rowptrs[i+1]--;

	for (NeuronID * c = rowbegin ; c <= rowend ; ++c) 
	{
		*get_ptr(c-colinds) *= value;
	}
}

template <typename T>
void ComplexMatrix<T>::scale_all(T value)
{
	for ( AurynLong i = 0 ; i < n_nonzero ; ++i ) 
		scale_data( i , value );
}

template <typename T>
void ComplexMatrix<T>::set_row(NeuronID i, T value)
{
	NeuronID * rowbegin = rowptrs[i];
	NeuronID * rowend = rowptrs[i+1]--;

	for (NeuronID * c = rowbegin ; c <= rowend ; ++c) 
	{
		*get_ptr(c-colinds) = value;
	}
}

template <typename T>
void ComplexMatrix<T>::set_all(T value, StateID z)
{
	get_state_vector(z)->set_all(value);
}

template <typename T>
void ComplexMatrix<T>::scale_col(NeuronID j, T value)
{
	for ( AurynLong i = 0 ; i < n_nonzero ; ++i ) {
		if ( colinds[i] == j ) scale_data(i,value);
	}
}

template <typename T>
double ComplexMatrix<T>::sum_col(NeuronID j)
{
	double sum = 0;
	for ( AurynLong i = 0 ; i < n_nonzero ; ++i ) {
		if ( colinds[i] == j ) 
			sum += *(get_data_begin()+i);
	}
	return sum;
}

template <typename T>
void ComplexMatrix<T>::set_col(NeuronID j, T value)
{
	for ( AurynLong i = 0 ; i < n_nonzero ; ++i ) {
		if ( colinds[i] == j ) set_data(i,value);
	}
}

template <typename T>
NeuronID * ComplexMatrix<T>::get_ind_begin()
{
	return rowptrs[0];
}

template <typename T>
NeuronID * ComplexMatrix<T>::get_row_begin(NeuronID i)
{
	return rowptrs[i];
}

template <typename T>
AurynLong ComplexMatrix<T>::get_row_begin_index(NeuronID i)
{
	return rowptrs[i]-rowptrs[0];
}

template <typename T>
NeuronID * ComplexMatrix<T>::get_row_end(NeuronID i)
{
	return rowptrs[i+1];
}

template <typename T>
AurynLong ComplexMatrix<T>::get_row_end_index(NeuronID i)
{
	return rowptrs[i+1]-rowptrs[0];
}


template <typename T>
NeuronID ** ComplexMatrix<T>::get_rowptrs()
{
	return rowptrs;
}

template <typename T>
T * ComplexMatrix<T>::get_data_begin(StateID z)
{
	return get_data_ptr((AurynLong)0,z);
}

template <typename T>
T * ComplexMatrix<T>::get_data_end(StateID z)
{
	return get_data_ptr(get_nonzero(),z);
}


template <typename T>
AurynDouble ComplexMatrix<T>::get_fill_level()
{
	return 1.*n_nonzero/get_datasize();
}

template <typename T>
NeuronID ComplexMatrix<T>::get_m_rows()
{
	return m_rows;
}

template <typename T>
NeuronID ComplexMatrix<T>::get_n_cols()
{
	return n_cols;
}


template <typename T>
void ComplexMatrix<T>::print()
{
	std::cout << get_nonzero() << " elements in sparse matrix:" << std::endl;
	for (NeuronID i = 0 ; i < m_rows ; ++i) {
		for (NeuronID * r = get_row_begin(i) ; r != get_row_end(i) ; ++r ) {
			std::cout << i << " " << *r << " " << *get_ptr(r-colinds) << "\n"; 
			// TODO not dumping the other states yet
		}
	}
}

template <typename T>
double ComplexMatrix<T>::mean()
{
	double sum = 0;
	for (NeuronID i = 0 ; i < get_nonzero() ; ++i) {
		sum += *get_ptr(i);
	}
	return sum/get_nonzero();
}

template <typename T>
void ComplexMatrix<T>::state_set_all(T * x, T value)
{
	for (T * iter=x;
			 iter!=x+get_nonzero();
			 ++iter ) {
		*iter = value;
	}
}

template <typename T>
void ComplexMatrix<T>::state_saxpy(T a, T * x, T * y)
{
	for (AurynLong i = 0 ; i < get_nonzero() ; ++i ) {
		y[i] = a*x[i]+y[i];
	}
}

template <typename T>
void ComplexMatrix<T>::state_mul(T * x, T * y)
{
	for (AurynLong i = 0 ; i < get_nonzero() ; ++i ) {
		y[i] = x[i]*y[i];
	}
}

template <typename T>
void ComplexMatrix<T>::state_add(T * x, T * y)
{
	for (AurynLong i = 0 ; i < get_nonzero() ; ++i ) {
		y[i] = x[i]+y[i];
	}
}

template <typename T>
void ComplexMatrix<T>::state_sub(T * x, T * y, T * res)
{
	for (AurynLong i = 0 ; i < get_nonzero() ; ++i ) {
		res[i] = x[i]-y[i];
	}
}

template <typename T>
void ComplexMatrix<T>::state_sub(T * x, T * y)
{
	state_sub(x,y,y);
}

template <typename T>
void ComplexMatrix<T>::state_scale(T a, T * y)
{
	for (AurynLong i = 0 ; i < get_nonzero() ; ++i ) {
		y[i] = a*y[i];
	}
}

template <typename T>
void ComplexMatrix<T>::state_add_const(T a, T * y)
{
	for (AurynLong i = 0 ; i < get_nonzero() ; ++i ) {
		y[i] = a+y[i];
	}
}

template <typename T>
void ComplexMatrix<T>::state_clip(T * x, T a, T b)
{
	for (AurynLong i = 0 ; i < get_nonzero() ; ++i ) {
		if ( x[i] < a ) x[i] = a; 
		else if ( x[i] > b ) x[i] = b; 
	}
}

template <typename T>
T * ComplexMatrix<T>::state_get_data_ptr(T * x, NeuronID i)
{
	return x+i;
}

template <typename T>
T ComplexMatrix<T>::get_value(NeuronID * r)
{
	return get_data_begin()[r-get_ind_begin()];
}

template <typename T>
T * ComplexMatrix<T>::get_value_ptr(NeuronID i)
{
	return &get_data_begin()[i];
}

template <typename T>
T * ComplexMatrix<T>::get_value_ptr(NeuronID * r)
{
	return &get_data_begin()[r-get_ind_begin()];
}

template <typename T>
NeuronID ComplexMatrix<T>::get_data_offset(NeuronID * r)
{
	return r-get_ind_begin();
}

}

#endif /*COMPLEXMATRIX_H_*/

