/* 
*
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

#ifndef AURYNVERSION_H_
#define AURYNVERSION_H_

#include <string>
#include <iostream>
#include "auryn_definitions.h"

namespace auryn {

	/*! \brief Container class providing Auryn version number */
	class AurynVersion { 
		public:
			static int version;
			static int subversion;
			static int revision_number;
			static NeuronID tag_binary_spike_monitor;
			static AurynState tag_binary_state_monitor;
			static std::string revision_suffix;
			static std::string git_describe;
		
		AurynVersion() {};
		virtual ~AurynVersion() {};

		std::string get_version_string()
		{
			std::stringstream oss;
			oss << version
				<< "."
				<< subversion;

			if ( revision_number ) {
				oss << "."
				<< revision_number;
			}

			oss << revision_suffix;

			return oss.str();
		}
	};

}


#endif /*AURYNVERSION_H_*/
