/* 
* Copyright 2014 Friedemann Zenke
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
*/

#include <iostream>
#include "SimpleMatrix.cpp"

using namespace std;

int main() {
	int size = 3;
	SimpleMatrix<double> mat(size,size,size*size);

	// cout << "fill" <<  endl;
	// for (int x = 0 ; x<size*size ; x += 1)
	// {
	// 	mat.push_back(x/size,x%size,x+10);
	// }
	// mat.push_back(0,1,1.);
	// mat.push_back(1,1,1.);
	mat.push_back(2,1,1.);

	mat.fill_zeros();

	NeuronID ** ptr = mat.get_rowptrs();
	for ( int i = 0 ; i <size+1 ; ++i ) cout <<  ptr[i] << "  ";
	cout << endl;
	double * dat = mat.get_data_begin();
	for ( int i = 0 ; i <size*size ; ++i ) cout << ptr[0][i] << ": " << dat[i] << "  ";
	cout << endl;


	cout << "read" <<  endl;

	for (int i = 0 ; i < size ; ++i)
	{
		for ( int j = 0 ; j < size ; ++j )
		{
			cout << i << " " << j << " " << mat.get(i,j) << endl;
			// cout << i << " " << j << " ptr " << mat.get_ptr(i,j) << endl;
		}
	}
}
