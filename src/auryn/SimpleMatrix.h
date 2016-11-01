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

#ifndef SIMPLEMATRIX_H_
#define SIMPLEMATRIX_H_

#include "auryn_global.h"
#include <string.h>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

namespace auryn {

/*! \brief Template for a sparse matrix with row major ordering and fast access of rows.
 *
 * This matrix class implements a sparse matrix that allows fast access of all elements of one row. It provides row interators to efficiently propagate spikes.
 * Memory has to be reserved when the class is defined and elements can only be inserted row by row starting from "left to right". This scheme enables related data fields to reside in memory next to each other.
 */
template <typename T>
class SimpleMatrix 
{
private:
	friend class boost::serialization::access;
	template<class Archive>
	void save(Archive & ar, const unsigned int version) const
		{
			ar & m_rows;
			ar & n_cols;
			ar & datasize;
			ar & n_nonzero;
			for (NeuronID i = 0 ; i < m_rows ; ++i) {
				for (NeuronID * r = rowptrs[i] ; r < rowptrs[i+1] ; ++r ) {
					ar & i;
					ar & *r;
					ar & coldata[r-colinds];
				}
			}
		}
	template<class Archive>
	void load(Archive & ar, const unsigned int version)
		{
			ar & m_rows;
			ar & n_cols;
			ar & datasize;
			resize_buffer_and_clear(datasize);

			AurynLong nnz;
			ar & nnz;

			for (AurynLong c = 0 ; c < nnz ; ++c) {
				NeuronID i,j;
				T val;
				ar & i;
				ar & j;
				ar & val;
				push_back(i,j,val);
			}
		}
	BOOST_SERIALIZATION_SPLIT_MEMBER()

	/*! Dimensions of matrix */
	NeuronID m_rows,n_cols;
	/*! Current row and col for filling. Should be in the last corner before use. */
	NeuronID current_row,current_col;
	/*! The number of actual nonzero elements. */
	AurynLong n_nonzero;
	/*! The size of data reserved and therefore the maximum number of non-zero elements. */
	AurynLong datasize;
protected:
	/*! Array that holds the begin addresses of column indices */
	NeuronID ** rowptrs;
	/*! Array that holds the column indices of non-zero elements */
	NeuronID * colinds;
	/*! Array that holds the data values of the non-zero elements */
	T * coldata; 

	/*! Default initializiation called by the constructor. */
	void init(NeuronID m, NeuronID n, AurynLong size=256);
	void free();

public:
	SimpleMatrix();
	SimpleMatrix(SimpleMatrix * mat);
	SimpleMatrix(NeuronID rows, NeuronID cols);
	SimpleMatrix(NeuronID rows, NeuronID cols, AurynLong size);
	virtual ~SimpleMatrix();

	void clear();

	/*! Resize buffer
	 *  Allocates a new buffer of size and copies the 
	 *  old buffers before freeing the memory */
	void resize_buffer(AurynLong newsize);

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
	void push_back(const NeuronID i, const NeuronID j, const T value);
	void copy(SimpleMatrix * mat);
	void set_data(AurynLong i, T value);
	void scale_data(AurynLong i, T value);
	/*! Gets the matching data ptr for a given index i*/
	T * get_data_ptr(const AurynLong i);
	/*! Gets the matching data entry for a given index i*/
	T get_data(const AurynLong i);
	/*! Gets the matching data ptr for a given index pointer */
	T * get_data_ptr(const NeuronID * ind_ptr);
	/*! Gets the matching data value for a given index pointer */
	T get_data(const NeuronID * ind_ptr);
	void fill_zeros();
	AurynDouble get_fill_level();
	T get(NeuronID i, NeuronID j);
	bool exists(NeuronID i, NeuronID j);
	/*! Returns the pointer to a particular element */
	T * get_ptr(NeuronID i, NeuronID j);
	/*! Returns the pointer to a particular element given 
	 * its position in the data array. */
	T * get_ptr(AurynLong data_index);
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
	AurynLong get_datasize();
	AurynLong get_nonzero();
	void print();
	double mean();
	NeuronID * get_ind_begin();
	NeuronID * get_row_begin(NeuronID i);
	AurynLong get_row_begin_index(NeuronID i);
	NeuronID * get_row_end(NeuronID i);
	AurynLong get_row_end_index(NeuronID i);
	NeuronID get_m_rows();
	NeuronID get_n_cols();
	NeuronID ** get_rowptrs();
	T * get_data_begin();
	/*! \brief Returns value behind last element in data array corresponding to a nonzero value. */
	T * get_data_end();
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
T * SimpleMatrix<T>::get_data_ptr(AurynLong i)
{
	return coldata+i;
}

template <typename T>
T SimpleMatrix<T>::get_data(AurynLong i)
{
	return coldata[i];
}

template <typename T>
T * SimpleMatrix<T>::get_data_ptr(const NeuronID * ind_ptr) 
{
	size_t ptr_offset = ind_ptr-get_ind_begin();
	return coldata+ptr_offset;
}

template <typename T>
T SimpleMatrix<T>::get_data(const NeuronID * ind_ptr) 
{
	return *(get_data_ptr(ind_ptr));
}

template <typename T>
void SimpleMatrix<T>::set_data(AurynLong i, T value)
{
	if (i<datasize)
		coldata[i] = value;
}

template <typename T>
void SimpleMatrix<T>::scale_data(AurynLong i, T value)
{
	if (i<datasize)
		coldata[i] *= value;
}


template <typename T>
void SimpleMatrix<T>::clear()
{
	current_row = 0;
	current_col = 0;
	n_nonzero = 0;
	rowptrs[0] = colinds;
	rowptrs[1] = colinds;
}

template <typename T>
void SimpleMatrix<T>::init(NeuronID m, NeuronID n, AurynLong size)
{
	m_rows = m;
	n_cols = n;
	datasize = size;
	rowptrs = new NeuronID * [m_rows+1];
	colinds = new NeuronID [datasize];
	coldata = new T [datasize];
	clear();
}

template <typename T>
void SimpleMatrix<T>::resize_buffer(AurynLong newsize)
{
	AurynLong oldsize = datasize;
	AurynLong copysize = std::min(oldsize,newsize);

	// std::cout << " copy colinds " << std::endl;

	NeuronID * new_colinds = new NeuronID [newsize];
	std::copy(colinds, colinds+copysize, new_colinds);

	// std::cout << " update rowptrs " << std::endl;

	// update rowpointers
	ptrdiff_t offset = new_colinds-colinds;
	for ( NeuronID i = 0 ; i < m_rows+1 ; ++i ) {
		rowptrs[i] += offset;
	}
	
	// std::cout << " delete old colinds " << std::endl;
	// FIXME I get regular crashes in this line:
	//  invalid pointer: 0x0000000001d954c0 ***
	delete [] colinds;

	// std::cout << " replace colinds " << std::endl;
	colinds = new_colinds;

	// std::cout << " create new data array " << std::endl;
	T * new_coldata = new T [newsize];
	std::copy(coldata, coldata+copysize, new_coldata);
	delete [] coldata;
	coldata = new_coldata;

	datasize = newsize;
	// std::cout << " done " << std::endl;
}

template <typename T>
void SimpleMatrix<T>::resize_buffer_and_clear(AurynLong size)
{
	// free();
	// init(m_rows, n_cols, size);
	resize_buffer(size);
	clear();
}

template <typename T>
void SimpleMatrix<T>::prune()
{
	if ( get_datasize() > get_nonzero() )
		resize_buffer(get_nonzero());
}

template <typename T>
SimpleMatrix<T>::SimpleMatrix()
{
	init(1,1);
}

template <typename T>
SimpleMatrix<T>::SimpleMatrix(SimpleMatrix * mat)
{
	init(mat->get_m_rows(),mat->get_n_cols(),mat->get_nonzero());
	copy(mat);
}



template <typename T>
SimpleMatrix<T>::SimpleMatrix(NeuronID rows, NeuronID cols)
{
	init(rows,cols);
}

template <typename T>
SimpleMatrix<T>::SimpleMatrix(NeuronID rows, NeuronID cols, AurynLong datasize)
{
	init(rows,cols,datasize);
}

template <typename T>
void SimpleMatrix<T>::free()
{
	delete [] rowptrs;
	delete [] colinds;
	delete [] coldata;
}

template <typename T>
void SimpleMatrix<T>::copy(SimpleMatrix * mat)
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
SimpleMatrix<T>::~SimpleMatrix()
{
	free();
}

template <typename T>
void SimpleMatrix<T>::push_back(NeuronID i, NeuronID j, T value)
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
		if ( n_nonzero >= datasize ) throw AurynMatrixBufferException();
		*(rowptrs[i+1]) = j; // write last j to end of index array
		coldata[rowptrs[i+1]-colinds] = value; // write value to end of data array
		++rowptrs[i+1]; //increment end by one
		rowptrs[m_rows] = rowptrs[i+1]; // last (m_row+1) marks end of last row
		n_nonzero++;
	} else {
		throw AurynMatrixPushBackException();
	}
}

template <typename T>
AurynLong SimpleMatrix<T>::get_datasize()
{
	return datasize;
}


template <typename T>
AurynLong SimpleMatrix<T>::get_nonzero()
{
	return n_nonzero;
}

template <typename T>
void SimpleMatrix<T>::fill_zeros()
{
	for ( NeuronID i = current_row ; i < m_rows-1 ; ++i )
	{
		rowptrs[i+2] = rowptrs[i+1];  // save value of last element
	}
	current_row = get_m_rows();
}


template <typename T>
T SimpleMatrix<T>::get(NeuronID i, NeuronID j)
{
	return *get_ptr(i,j);
}

template <typename T>
bool SimpleMatrix<T>::exists(NeuronID i, NeuronID j)
{
	if ( get_ptr(i,j) == NULL )
		return false;
	else 
		return true;
}

template <typename T>
T * SimpleMatrix<T>::get_ptr(NeuronID i, NeuronID j)
{
	// check bounds
	if ( !(i < m_rows && j < n_cols) ) return NULL;

	// perform binary search
	NeuronID * lo = rowptrs[i];
	NeuronID * hi = rowptrs[i+1];
	NeuronID * c  = hi;

	if ( lo >= hi ) // no elements/targets in this row
		return NULL; 
	
	while ( lo < hi ) {
		c = lo + (hi-lo)/2;
		if ( *c < j ) lo = c+1;
		else hi = c;
		//std::cout << i << ":" << j << "   " << *lo << ":" << *hi << endl;
	}
	
	if ( *lo == j ) {
		return coldata+(lo-colinds);
	}

	return NULL; 
}

template <typename T>
T * SimpleMatrix<T>::get_ptr(AurynLong data_index)
{
	return &coldata[data_index];
}

template <typename T>
T SimpleMatrix<T>::get_value(AurynLong data_index)
{
	return coldata[data_index];
}

template <typename T>
void SimpleMatrix<T>::add_value(AurynLong data_index, T value)
{
	coldata[data_index] += value;
}

template <typename T>
NeuronID SimpleMatrix<T>::get_colind(AurynLong data_index)
{
	return colinds[data_index];
}

template <typename T>
bool SimpleMatrix<T>::set(NeuronID i, NeuronID j, T value)
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
void SimpleMatrix<T>::scale_row(NeuronID i, T value)
{
	NeuronID * rowbegin = rowptrs[i];
	NeuronID * rowend = rowptrs[i+1]--;

	for (NeuronID * c = rowbegin ; c <= rowend ; ++c) 
	{
		coldata[c-colinds] *= value;
	}
}

template <typename T>
void SimpleMatrix<T>::scale_all(T value)
{
	for ( AurynLong i = 0 ; i < n_nonzero ; ++i ) 
		scale_data( i , value );
}

template <typename T>
void SimpleMatrix<T>::set_row(NeuronID i, T value)
{
	NeuronID * rowbegin = rowptrs[i];
	NeuronID * rowend = rowptrs[i+1]--;

	for (NeuronID * c = rowbegin ; c <= rowend ; ++c) 
	{
		coldata[c-colinds] = value;
	}
}

template <typename T>
void SimpleMatrix<T>::set_all(T value)
{
	for ( AurynLong i = 0 ; i < n_nonzero ; ++i ) 
		set_data( i , value );
}

template <typename T>
void SimpleMatrix<T>::scale_col(NeuronID j, T value)
{
	for ( AurynLong i = 0 ; i < n_nonzero ; ++i ) {
		if ( colinds[i] == j ) scale_data(i,value);
	}
}

template <typename T>
double SimpleMatrix<T>::sum_col(NeuronID j)
{
	double sum = 0;
	for ( AurynLong i = 0 ; i < n_nonzero ; ++i ) {
		if ( colinds[i] == j ) 
			sum += *(get_data_begin()+i);
	}
	return sum;
}

template <typename T>
void SimpleMatrix<T>::set_col(NeuronID j, T value)
{
	for ( AurynLong i = 0 ; i < n_nonzero ; ++i ) {
		if ( colinds[i] == j ) set_data(i,value);
	}
}

template <typename T>
NeuronID * SimpleMatrix<T>::get_ind_begin()
{
	return rowptrs[0];
}

template <typename T>
NeuronID * SimpleMatrix<T>::get_row_begin(NeuronID i)
{
	return rowptrs[i];
}

template <typename T>
AurynLong SimpleMatrix<T>::get_row_begin_index(NeuronID i)
{
	return rowptrs[i]-rowptrs[0];
}

template <typename T>
NeuronID * SimpleMatrix<T>::get_row_end(NeuronID i)
{
	return rowptrs[i+1];
}

template <typename T>
AurynLong SimpleMatrix<T>::get_row_end_index(NeuronID i)
{
	return rowptrs[i+1]-rowptrs[0];
}


template <typename T>
NeuronID ** SimpleMatrix<T>::get_rowptrs()
{
	return rowptrs;
}

template <typename T>
T * SimpleMatrix<T>::get_data_begin()
{
	return coldata;
}

template <typename T>
T * SimpleMatrix<T>::get_data_end()
{
	return coldata+get_nonzero();
}


template <typename T>
AurynDouble SimpleMatrix<T>::get_fill_level()
{
	return 1.*n_nonzero/datasize;
}

template <typename T>
NeuronID SimpleMatrix<T>::get_m_rows()
{
	return m_rows;
}

template <typename T>
NeuronID SimpleMatrix<T>::get_n_cols()
{
	return n_cols;
}

template <typename T>
void SimpleMatrix<T>::print()
{
	for (NeuronID i = 0 ; i < m_rows ; ++i) {
		for (NeuronID * r = get_row_begin(i) ; r != get_row_end(i) ; ++r ) {
			std::cout << i << " " << *r << " " << coldata[r-colinds] << "\n";
		}
	}
}

template <typename T>
double SimpleMatrix<T>::mean()
{
	double sum = 0;
	for (NeuronID i = 0 ; i < get_nonzero() ; ++i) {
		sum += coldata[i];
	}
	return sum/get_nonzero();
}

template <typename T>
T SimpleMatrix<T>::get_value(NeuronID i)
{
	return get_data_begin()[i];
}

template <typename T>
T SimpleMatrix<T>::get_value(NeuronID * r)
{
	return get_data_begin()[r-get_ind_begin()];
}

template <typename T>
T * SimpleMatrix<T>::get_value_ptr(NeuronID i)
{
	return &get_data_begin()[i];
}

template <typename T>
T * SimpleMatrix<T>::get_value_ptr(NeuronID * r)
{
	return &get_data_begin()[r-get_ind_begin()];
}

template <typename T>
NeuronID SimpleMatrix<T>::get_data_offset(NeuronID * r)
{
	return r-get_ind_begin();
}

}


#endif /*SIMPLEMATRIX_H_*/

