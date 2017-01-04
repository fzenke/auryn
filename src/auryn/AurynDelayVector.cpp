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

#include "AurynDelayVector.h"

using namespace auryn;



AurynDelayVector::AurynDelayVector( NeuronID n, unsigned int delay_buffer_size ) : AurynVectorFloat(n)
{
	memory_pos_ = 0;
	delay_buffer_size_ = delay_buffer_size;
	for ( unsigned int i = 0 ; i < delay_buffer_size_ ; ++i ) {
		AurynVectorFloat * vec = new AurynVectorFloat(size);
		memory.push_back(vec);
	}
}

AurynDelayVector::~AurynDelayVector(  ) 
{
	for ( unsigned int i = 0 ; i < delay_buffer_size_ ; ++i ) {
		delete memory[i];
	}
}

void AurynDelayVector::resize( NeuronID new_size ) 
{
	super::resize(new_size);
	for ( unsigned int i = 0 ; i < delay_buffer_size_ ; ++i ) {
		memory[i]->resize(new_size);
	}
}

void AurynDelayVector::advance(  ) 
{
	// TODO could improve performance of the whole class if instead of copying we just
	// juggle the data pointers ...
	memory[memory_pos_]->copy(this);
	memory_pos_ = (memory_pos_+1)%delay_buffer_size_;
}


AurynVectorFloat * AurynDelayVector::mem_get_vector( int delay ) 
{
	if ( delay == 0 ) return this;
	int pos = (delay_buffer_size_+memory_pos_-delay)%delay_buffer_size_;
	if ( delay < 0 ) pos = memory_pos_;
	return memory[pos];
}

AurynFloat AurynDelayVector::mem_get( NeuronID i, int delay ) 
{
	return mem_get_vector(delay)->get(i);
}

AurynVectorFloat * AurynDelayVector::mem_ptr( int delay ) 
{
	if ( delay == 0 ) return this;
	int pos = (delay_buffer_size_+memory_pos_-delay)%delay_buffer_size_;
	if ( delay < 0 ) pos = memory_pos_;
	return memory[pos];
}
