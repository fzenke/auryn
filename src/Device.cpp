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

#include "Device.h"

using namespace auryn;

int Device::device_id_count = 0;

void Device::init()
{
	device_id  = device_id_count++;
	std::stringstream default_name;
	default_name << "Device" << get_id();
	set_name(default_name.str());
	active = true;
}

Device::Device( )
{
	init();
}

Device::~Device()
{
}

void Device::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
}

void Device::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
}

void Device::flush()
{
}

void Device::set_name( std::string str ) 
{
	device_name = str;
}

std::string Device::get_name()
{
	return device_name;
}

int Device::get_id()
{
	return device_id;
}

