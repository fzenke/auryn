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

#include "Monitor.h"

void Monitor::init(string filename)
{
	if ( filename.empty() ) return; // stimulators do not necessary need an outputfile

	fname = filename;
	active = true;

	outfile.open(filename.c_str(),ios::out);
	if (!outfile) {
	  stringstream oss;
	  oss << "Can't open output file " << filename;
	  logger->msg(oss.str(),ERROR);
	  exit(1);
	}

	outfile << setiosflags(ios::fixed) << setprecision(log(dt)/log(10)+1);
}

Monitor::Monitor()
{
}

Monitor::Monitor(string filename)
{
	init(filename);
}

Monitor::~Monitor()
{
	free();
}

void Monitor::free()
{
	outfile.close();
}

