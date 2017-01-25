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

#include "RealTimeMonitor.h"

using namespace auryn;

RealTimeMonitor::RealTimeMonitor(std::string filename, AurynDouble start, AurynDouble stop) : Monitor(filename)
{
	auryn::sys->register_device(this);

	t_start = start/auryn_timestep;
	t_stop = stop/auryn_timestep;

	ptime_offset = boost::posix_time::microsec_clock::local_time();

#ifdef AURYN_CODE_USE_MPI
	if (auryn::sys->get_com()->rank() == 0) {
		std::string sendstring = boost::posix_time::to_iso_string(ptime_offset);
		broadcast(*auryn::sys->get_com(), sendstring, 0);
	} else {
		std::string timestring;
		broadcast(*auryn::sys->get_com(), timestring , 0);
		ptime_offset = boost::posix_time::from_iso_string(timestring);
	}
#endif // AURYN_CODE_USE_MPI
}

RealTimeMonitor::~RealTimeMonitor()
{
}

void RealTimeMonitor::execute()
{
	if ( t_stop > auryn::sys->get_clock() && t_start < auryn::sys->get_clock() ) {
		boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
		boost::posix_time::time_duration diff = now - ptime_offset;
		outfile << (auryn::sys->get_clock()) << " " << diff.total_microseconds() << "\n";
	}
}
