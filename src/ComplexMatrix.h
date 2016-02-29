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
			ar & z_values;
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
			for (StateID z = 0; z < z_values ; ++z ) {
				for (AurynLong i = 0 ; i < n_nonzero ; ++i) {
					ar & elementdata[i+z*statesize];
				}
			}
		}
	template<class Archive>
	void load(Archive & ar, const unsigned int version)
		{
			ar & m_rows;
			ar & n_cols;
			ar & z_values;
			ar & current_row;
			ar & current_col;
			ar & statesize;
			ar & n_nonzero;

			// allocate necessary memory
			resize_buffer(statesize);

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
			for (StateID z = 0; z < z_values ; ++z ) {
				for (AurynLong i = 0 ; i < n_nonzero ; ++i) {
					ar & elementdata[i+z*statesize];
				}
			}
		}
	BOOST_SERIALIZATION_SPLIT_MEMBER()

	/*! Dimensions of matrix */
	NeuronID m_rows,n_cols;
	/*! z-dimension of matrix */
	NeuronID z_values;
	/*! Current row and col for filling. Should be in the last corner before use. */
	NeuronID current_row, current_col;
	/*! The number of actual nonzero elements. */
	AurynLong n_nonzero;
	/*! The size of data reserved and therefore the maximum number of non-zero elements. */
	AurynLong statesize;
protected:
	/*! Array that holds the begin addresses of column indices */
	NeuronID ** rowptrs;
	/*! Array that holds the column indices of non-zero elements */
	NeuronID * colinds;
	/*! Array that holds the data values of the non-zero elements */
	T * elementdata; 

	/*! Default initializiation called by the constructor. */
	void init(NeuronID m, NeuronID n, AurynLong size, NeuronID z );
	void free();

public:
	ComplexMatrix();
	ComplexMatrix(ComplexMatrix * mat);
	ComplexMatrix(NeuronID rows, NeuronID cols, AurynLong size=256, NeuronID values=1 );
	virtual ~ComplexMatrix();

	void clear();

	/*! Resize buffer
	 *  Allocates a new buffer of size and copies the 
	 *  old buffers before freeing the memory */
	void resize_buffer(AurynLong size);

	/*! Resizes buffer and clears the matrix. This saves
	 *  to copy all the data. */
	void resize_buffer_and_clear(AurynLong size);

	/*! Prunes the reserved memory such that datasize=get_nonzero()
	 *  Note that this is an expensive operation because it uses
	 *  resize_buffer(size) and it should be used sparsely. */
	void prune();

	/*! Add non-zero element in row/col order. 
	 * \param i row index where to insert element 
	 * \param j col index where to insert element
	 * \param value to insert
	 * \throw AurynMatrixBufferException */
	void push_back(NeuronID i, NeuronID j, T value);
	void copy(ComplexMatrix * mat);
	void set_data(AurynLong i, T value);
	void scale_data(AurynLong i, T value);
	/*! Gets the matching data entry for a given index i and state z*/
	T get_data(AurynLong i, StateID z=0);
	/*! Gets the matching data ptr for a given index i and state z*/
	T * get_data_ptr(AurynLong i, StateID z=0);
	/*! Gets the matching data ptr for a given index pointer and state z*/
	T * get_data_ptr(const NeuronID * ind_ptr, StateID z=0);
	/*! Gets the matching data value for a given index pointer and state z*/
	T get_data(const NeuronID * ind_ptr, StateID z=0);
	void fill_zeros();
	AurynDouble get_fill_level();
	T get(NeuronID i, NeuronID j, NeuronID z=0);
	bool exists(NeuronID i, NeuronID j);
	/*! Returns the pointer to a particular element */
	T * get_ptr(NeuronID i, NeuronID j);

	T * get_ptr(NeuronID i, NeuronID j, NeuronID z);
	/*! Returns the pointer to a particular element given 
	 * its position in the data array. */
	T * get_ptr(AurynLong data_index);

	/*! Returns data index to a particular element specifed by i and j */
	AurynLong get_data_index(NeuronID i, NeuronID j, NeuronID z=0);

	/*! \brief Returns data index to a particular element specifed by an index pointer */
	AurynLong get_data_index(const NeuronID * ind_ptr);

	/*! \brief Returns data index to a particular element specifed by a data pointer */
	AurynLong get_data_index(const T * ptr);

	/* Methods concerning synaptic state vectors. */
	void set_num_synapse_states(StateID zsize);
	/*! Gets pointer for the first element of a given synaptic state vector */
	T * get_state_begin(StateID z=0);
	/*! Gets pointer for the element behind the last of a given synaptic state vector */
	T * get_state_end(StateID z=0);
	/*! Sets all values in state x to value. */
	void state_set_all(T * x, T value);
	/*! Computes a*x + y and stores result in y */
	void state_saxpy(T a, T * x, T * y);
	/*! Multiplies x and y and stores result in y */
	void state_mul(T * x, T * y);
	/*! Adds x and y and stores result in y */
	void state_add(T * x, T * y);
	/*! Computes x-y and stores result in y */
	void state_sub(T * x, T * y);
	/*! Computes x-y and stores result in res */
	void state_sub(T * x, T * y, T * res);
	/*! Scale state x by a. */
	void state_scale(T a, T * x);
	/*! Adds constant a to all values in x */
	void state_add_const(T a, T * x);
	/*! Clips state values to interval [a,b] */
	void state_clip(T * x, T a, T b);
	/*! Get data pointer for that state */
	T * state_get_data_ptr(T * x, NeuronID i);

	T get_value(AurynLong data_index);
	void add_value(AurynLong data_index, T value);
	NeuronID get_colind(AurynLong data_index);
	bool set(NeuronID i, NeuronID j, T value);
	/*! Sets all non-zero elements to value */
	void set_all(T value);
	/*! Sets all non-zero elements in row i to value */
	void set_row(NeuronID i, T value);
	/*! Scales all non-zero elements in row i to value */
	void scale_row(NeuronID i, T value);
	/*! Scales all non-zero elements */
	void scale_all(T value);
	/*! Sets all non-zero elements in col j to value. Due to ordering this is slow and the use of this functions is discouraged. */
	void set_col(NeuronID j, T value);
	/*! Scales all non-zero elements in col j to value. Due to ordering this is slow and the use of this functions is discouraged. */
	void scale_col(NeuronID j, T value);
	double sum_col(NeuronID j);
	/*! Returns datasize: number of possible entries */
	AurynLong get_datasize();
	/*! Same as datasize : number of possible entries */
	AurynLong get_statesize();
	/*! Returns statesize multiplied by number of states */
	AurynLong get_memsize();
	/*! Returns number of non-zero elements */
	AurynLong get_nonzero();
	/*! stdout dump of all elements -- for testing only. */
	void print();
	double mean();
	NeuronID * get_ind_begin();
	NeuronID * get_row_begin(NeuronID i);
	AurynLong get_row_begin_index(NeuronID i);
	NeuronID * get_row_end(NeuronID i);
	AurynLong get_row_end_index(NeuronID i);
	NeuronID get_m_rows();
	NeuronID get_n_cols();
	NeuronID get_z_values();
	NeuronID ** get_rowptrs();
	T * get_data_begin(const StateID z=0);
	/*! \brief Returns the data value corresponding to the element behind the last nonzero element. */
	T * get_data_end(const StateID z=0);
	/*! Returns the data value to an item that is i-th in the colindex array */
	T get_value(NeuronID i);
	/*! Returns the data value to an item that for pointer r pointing to the respective element in the index array */
	T get_value(NeuronID * r);
	/*! Returns pointer to the the data value to an item that is i-th in the colindex array */
	T * get_value_ptr(NeuronID i);
	/*! Returns the pointer to the data value to an item that for pointer r pointing to the respective element in the index array */
	T * get_value_ptr(NeuronID * r);
	NeuronID get_data_offset(NeuronID * r);
};

template <typename T>
T * ComplexMatrix<T>::get_data_ptr(const AurynLong i, const StateID z)
{
	return elementdata+z*get_datasize()+i;
}

template <typename T>
T ComplexMatrix<T>::get_data(const AurynLong i, const StateID z)
{
	return elementdata[z*get_datasize()+i];
}

template <typename T>
T * ComplexMatrix<T>::get_data_ptr(const NeuronID * ind_ptr, const StateID z) 
{
	size_t ptr_offset = ind_ptr-get_ind_begin();
	return elementdata+z*get_datasize()+ptr_offset;
}

template <typename T>
T ComplexMatrix<T>::get_data(const NeuronID * ind_ptr, StateID z) 
{
	return *(get_data_ptr(ind_ptr));
}


//FIXME the following functions still need to have added z value support
template <typename T>
void ComplexMatrix<T>::set_data(AurynLong i, T value)
{
	if (i<statesize)
		elementdata[i] = value;
}

//FIXME the following functions still need to have added z value support
template <typename T>
void ComplexMatrix<T>::scale_data(AurynLong i, T value)
{
	if (i<statesize)
		elementdata[i] *= value;
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
void ComplexMatrix<T>::init(NeuronID m, NeuronID n, AurynLong size, NeuronID z)
{
	m_rows = m;
	n_cols = n;
	statesize = size; // assumed maximum number of connections
	z_values = z;
	

	rowptrs = new NeuronID * [m_rows+1];
	colinds = new NeuronID [get_datasize()];
	elementdata = new T [get_memsize()];
	clear();
}

template <typename T>
void ComplexMatrix<T>::resize_buffer(AurynLong size)
{
	statesize = size;

	NeuronID * new_colinds = new NeuronID [get_datasize()];
	std::copy(colinds, colinds+get_nonzero(), new_colinds);

	// update rowpointers
	ptrdiff_t offset = new_colinds-colinds;
	for ( NeuronID i = 0 ; i < m_rows+1 ; ++i ) {
		rowptrs[i] += offset;
	}
	
	delete [] colinds;
	colinds = new_colinds;

	T * new_elementdata = new T [get_memsize()];
	std::copy(elementdata, elementdata+get_datasize(), new_elementdata);
	delete [] elementdata;
	elementdata = new_elementdata;
}

template <typename T>
void ComplexMatrix<T>::resize_buffer_and_clear(AurynLong size)
{
	free();
	init(m_rows, n_cols, size, z_values );
}

template <typename T>
void ComplexMatrix<T>::prune()
{
	if ( get_datasize() > get_nonzero() )
		resize_buffer(get_nonzero());
}

template <typename T>
ComplexMatrix<T>::ComplexMatrix()
{
	init(1,1,2,1);
}

template <typename T>
ComplexMatrix<T>::ComplexMatrix(ComplexMatrix * mat)
{
	init(mat->get_m_rows(), mat->get_n_cols(), mat->get_nonzero(), mat->get_z_values() );
	copy(mat);
}

template <typename T>
ComplexMatrix<T>::ComplexMatrix(NeuronID rows, NeuronID cols, AurynLong statesize, NeuronID values )
{
	init(rows, cols, statesize, values);
}

template <typename T>
void ComplexMatrix<T>::free()
{
	delete [] rowptrs;
	delete [] colinds;
	delete [] elementdata;
}

template <typename T>
void ComplexMatrix<T>::copy(ComplexMatrix * mat)
{
	if ( get_m_rows() != mat->get_m_rows() || get_n_cols() != mat->get_n_cols() )
		throw AurynMatrixDimensionalityException();

	clear();

	for ( NeuronID i = 0 ; i < mat->get_m_rows() ; ++i ) {
		for ( NeuronID * r = mat->get_row_begin(i) ; r != mat->get_row_end(i) ; ++r ) {
			push_back(i,*r,mat->get_value(r));
		}
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
	if (i >= current_row && j >= current_col) {
		if ( n_nonzero >= get_datasize() ) throw AurynMatrixBufferException();
		*(rowptrs[i+1]) = j; // write last j to end of index array
		elementdata[rowptrs[i+1]-colinds] = value; // write value to end of data array
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
	return statesize*z_values;
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
T ComplexMatrix<T>::get(NeuronID i, NeuronID j, NeuronID z)
{
	return *get_ptr(i,j,z);
}

template <typename T>
bool ComplexMatrix<T>::exists(NeuronID i, NeuronID j)
{
	if ( get_ptr(i,j) == NULL )
		return false;
	else 
		return true;
}

template <typename T>
void ComplexMatrix<T>::set_num_synapse_states(StateID zsize)
{
	z_values = zsize;
	resize_buffer(statesize);
}

template <typename T>
T * ComplexMatrix<T>::get_state_begin(NeuronID z)
{
	return get_ptr(get_datasize()*z);
}


template <typename T>
T * ComplexMatrix<T>::get_state_end(NeuronID z)
{
	return get_ptr(get_datasize()*(z+1));
}

template <typename T>
T * ComplexMatrix<T>::get_ptr(NeuronID i, NeuronID j, NeuronID z)
{
	return get_ptr(i,j)+get_datasize()*z;
}

template <typename T>
T * ComplexMatrix<T>::get_ptr(NeuronID i, NeuronID j)
{
	// check bounds
	if ( !(i < m_rows && j < n_cols) ) return NULL;

	// perform binary search
	NeuronID * lo = rowptrs[i];
	NeuronID * hi = rowptrs[i+1];
	NeuronID * c  = hi;
	
	while ( lo < hi ) {
		c = lo + (hi-lo)/2;
		if ( *c < j ) lo = c+1;
		else hi = c;
		//std::cout << i << ":" << j << "   " << *lo << ":" << *hi << endl;
	}
	
	if ( *lo == j ) {
		return elementdata+(lo-colinds);
	}

	return NULL; 
}

template <typename T>
AurynLong ComplexMatrix<T>::get_data_index(NeuronID i, NeuronID j, NeuronID z)
{
	return get_data_index( get_ptr(i,j) ) + z*statesize ;
}

template <typename T>
AurynLong ComplexMatrix<T>::get_data_index(const NeuronID * ind_ptr)
{
	return ind_ptr - get_ind_begin();
}

template <typename T>
AurynLong ComplexMatrix<T>::get_data_index(const T * ptr)
{
	return ptr - get_data_begin();
}

template <typename T>
T * ComplexMatrix<T>::get_ptr(AurynLong data_index)
{
	return &elementdata[data_index];
}

template <typename T>
T ComplexMatrix<T>::get_value(AurynLong data_index)
{
	return elementdata[data_index];
}

template <typename T>
void ComplexMatrix<T>::add_value(AurynLong data_index, T value)
{
	elementdata[data_index] += value;
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
		elementdata[c-colinds] *= value;
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
		elementdata[c-colinds] = value;
	}
}

template <typename T>
void ComplexMatrix<T>::set_all(T value)
{
	for ( AurynLong i = 0 ; i < n_nonzero ; ++i ) 
		set_data( i , value );
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
NeuronID ComplexMatrix<T>::get_z_values()
{
	return z_values;
}

template <typename T>
void ComplexMatrix<T>::print()
{
	for (NeuronID i = 0 ; i < m_rows ; ++i) {
		for (NeuronID * r = get_row_begin(i) ; r != get_row_end(i) ; ++r ) {
			std::cout << i << " " << *r << " " << elementdata[r-colinds] << "\n"; 
			// FIXME not dumping the other states yet
		}
	}
}

template <typename T>
double ComplexMatrix<T>::mean()
{
	double sum = 0;
	for (NeuronID i = 0 ; i < get_nonzero() ; ++i) {
		sum += elementdata[i];
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
T ComplexMatrix<T>::get_value(NeuronID i)
{
	return get_data_begin()[i];
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

