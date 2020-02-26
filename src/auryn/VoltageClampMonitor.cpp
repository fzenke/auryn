/* 
* Copyright 2014-2020 Friedemann Zenke
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

#include "VoltageClampMonitor.h"

using namespace auryn;

VoltageClampMonitor::VoltageClampMonitor(NeuronGroup * source, NeuronID id, std::string filename) : VoltageMonitor(source, id, filename)
{
	clamping_voltage = -70e-3;
	clamp_enabled = true;
}

VoltageClampMonitor::~VoltageClampMonitor()
{
}

void VoltageClampMonitor::execute()
{
	if ( clamp_enabled && auryn::sys->get_clock() < t_stop ) {
		AurynState * voltage = target_variable;
		AurynState pseudo_current = clamping_voltage-*voltage;
		// TODO a unitful quantity would be nice here ... 
		*voltage += pseudo_current;
		outfile << (auryn::sys->get_time()) << " " << pseudo_current << "\n";
	}
}


