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

#include "SpikeDelay.h"

using namespace auryn;

AurynTime * SpikeDelay::clock_ptr = NULL;

SpikeDelay::SpikeDelay( int delay )
{
	ndelay = delay;
	numSpikeAttributes = 0;
	delaybuf = new SpikeContainer * [ndelay] ;
	attribbuf = new AttributeContainer * [ndelay] ;
	for (int i = 0 ; i < ndelay ; ++i) {
		delaybuf[i] = new SpikeContainer( );
		attribbuf[i] = new AttributeContainer( );
	}
}

void SpikeDelay::set_delay( int delay ) 
{
	if ( delay == ndelay ) return;
	free();
	ndelay = delay;
	delaybuf = new SpikeContainer * [ndelay] ;
	attribbuf = new AttributeContainer * [ndelay] ;
	for (int i = 0 ; i < ndelay ; ++i) {
		delaybuf[i] = new SpikeContainer( );
		attribbuf[i] = new AttributeContainer( );
	}
}

int SpikeDelay::get_delay( ) 
{
	return ndelay;
}

void SpikeDelay::free()
{
	for (unsigned int i = 0 ; i < ndelay ; ++i) {
		delete delaybuf[i];
		delete attribbuf[i];
	}
	delete [] delaybuf;
	delete [] attribbuf;
}

void SpikeDelay::clear()
{
	for (unsigned int i = 0 ; i < ndelay ; ++i) {
		delaybuf[i]->clear();
		attribbuf[i]->clear();
	}
}

SpikeDelay::~SpikeDelay() 
{
	free();
}

SpikeContainer * SpikeDelay::get_spikes(unsigned int pos)
{
	return delaybuf[((*clock_ptr)+pos)%ndelay]; 
}

SpikeContainer * SpikeDelay::get_spikes_immediate()
{
	return get_spikes(0);
}

AttributeContainer * SpikeDelay::get_attributes(unsigned int pos)
{
	return attribbuf[((*clock_ptr)+pos)%ndelay]; 
}

AttributeContainer * SpikeDelay::get_attributes_immediate()
{
	return get_attributes(0);
}

void SpikeDelay::set_clock_ptr(AurynTime * clock) 
{
	SpikeDelay::clock_ptr = clock;
}

void SpikeDelay::insert_spike( NeuronID i, AurynTime ahead )
{
	get_spikes((*clock_ptr)+(1+ahead+ndelay))->push_back(i);
}

void SpikeDelay::push_back( NeuronID i )
{
	get_spikes_immediate()->push_back(i);
}

void SpikeDelay::push_back( SpikeContainer * sc )
{
	for ( NeuronID i = 0 ; i < sc->size() ; ++i ) {
		push_back(sc->at(i));
	}
}

int SpikeDelay::get_num_attributes( )
{
	return numSpikeAttributes;
}

void SpikeDelay::inc_num_attributes( int x )
{
	numSpikeAttributes += x;
}


void SpikeDelay::print()
{
	for ( unsigned int i = 0 ; i < ndelay ; ++i ) {
		SpikeContainer * spikes = get_spikes(i);
		if ( spikes->size() ) {
			std::cout << "slice " << i << ": ";
			for ( NeuronID k = 0 ; k < spikes->size() ; ++k ) {
				std::cout << spikes->at(k) << " ";
			}
			std::cout << std::endl;
		}
	}
}
