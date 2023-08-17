/* 
* Copyright 2014-2023 Friedemann Zenke
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

#include "Monitor.h"

using namespace auryn;

void Monitor::init(std::string filename)
{
	if ( filename.empty() ) { // generate default filename from device id
		fname = generate_filename();
	} else 
		fname = filename;

	active = true;
	open_output_file(fname);
}

Monitor::Monitor( std::string filename, std::string default_extension ) : Device()
{
	default_file_extension = default_extension;
	init(filename);
}

Monitor::Monitor( ) : Device()
{
	default_file_extension = "dat";
}

void Monitor::open_output_file(std::string filename)
{
	outfile.open( filename.c_str(), std::ios::out );
	if (!outfile) {
	  std::stringstream oss;
	  oss << "Can't open output file " << filename;
	  auryn::logger->msg(oss.str(),ERROR);
	  exit(1);
	}
}

std::string Monitor::generate_filename(std::string name_hint) 
{
	std::stringstream oss;
	oss << get_name() << name_hint;
	std::string tmpstr = oss.str(); 
	tmpstr.erase(std::remove(tmpstr.begin(),tmpstr.end(),' '),tmpstr.end());
	std::transform(tmpstr.begin(), tmpstr.end(), tmpstr.begin(), ::tolower);
	tmpstr = sys->fn(tmpstr, default_file_extension);
	// std::cout << tmpstr << std::endl;
	return tmpstr;
}

Monitor::~Monitor()
{
	free();
}

void Monitor::free()
{
	outfile.close();
}

void Monitor::flush()
{
	outfile.flush();
}


void Monitor::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	Device::virtual_serialize(ar, version);
}

void Monitor::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	Device::virtual_serialize(ar, version);
}
