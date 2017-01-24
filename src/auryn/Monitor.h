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

#ifndef MONITOR_H_
#define MONITOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Device.h"
#include "System.h"
#include <fstream>
#include <string>
#include <algorithm>

namespace auryn {

	class System;

	/*! \brief Abstract base class for all Monitor objects.
	 * 
	 * This Class constitutes the base class for all Monitors in Auryn. Per default it openes a single text file for writing (outfile) named by the name supplied in the constructor.
	 * Classes inheriting from Monitor have to implement the method execute. Unlike Checker objects execute returns void. Use Checker if you need the Monitor to be able to interrupt a run. 
	 */

	class Monitor : public Device
	{
	private:
		/*! Standard initializer to be called by the constructor */
		void init(std::string filename);

		/*! Standard free function to be called by the destructor - closes the file stream. */
		void free();

	protected:
		/*! Output filestream to be used in the derived classes */
		std::ofstream outfile;

		/*! Stores output filename */
		std::string fname;

		/*! Default extension */
		std::string default_file_extension;


		/*! Opens a text outputfile -- for binary files redefine this function in derived class. */
		virtual void open_output_file(std::string filename);


		/*! Functions necesssary for serialization and loading saving to netstate files. */
		virtual void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) ;
		virtual void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) ;
		
	public:
		/*! \brief Standard active switch */
		bool active;

		/*! \brief Flush to file */
		virtual void flush();

		/*! \brief Standard constructor with file name*/
		Monitor(std::string filename, std::string default_extension = "dat");

		/*! \brief Constructor which does not open a text file for output */
		Monitor();

		/*! \brief Generates a default filename from the device ID. */
		std::string generate_filename(std::string name_hint="");

		/*! \brief Standard destructor  */
		virtual ~Monitor();
	};

	extern System * sys;
	extern Logger * logger;
}

#endif /*MONITOR_H_*/
