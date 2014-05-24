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

#include "RealTimeMonitor.h"

RealTimeMonitor::RealTimeMonitor(string filename, AurynDouble start, AurynDouble stop) : Monitor(filename)
{
	sys->register_monitor(this);

	t_start = start/dt;
	t_stop = stop/dt;
	ptime_offset = boost::posix_time::microsec_clock::local_time();
}

RealTimeMonitor::~RealTimeMonitor()
{
}

void RealTimeMonitor::propagate()
{
	if ( t_stop > sys->get_clock() && t_start < sys->get_clock() ) {
		boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
		boost::posix_time::time_duration diff = now - ptime_offset;
		outfile << (sys->get_clock()) << " " << diff.total_milliseconds() << "\n";
	}
}
