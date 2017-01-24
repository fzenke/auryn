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

#ifndef DEVICE_H_
#define DEVICE_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include <fstream>
#include <string>

namespace auryn {

	class System;

	/*! \brief Abstract base class for all Device, Stimulator, etc objects.
	 * 
	 * Devices are executed after SpikingGroups and Connection objects in the Auryn duty cycle. 
	 * Most commonly a Device will be a Monitor, but it could be a stimulator or something else too.
	 */

	class Device
	{
	private:
		/*! Stores the unique device_id of this Device */
		int device_id;

		/*! Stores the current value of the gid count */
		static int device_id_count;

		/*! Functions necesssary for serialization and loading saving to netstate files. */
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			virtual_serialize(ar, version);
		}


	protected:
		/*! \brief Identifying name for device */
		std::string device_name;

		/*! \brief Standard initializer to be called by the constructor */
		void init();

		/* Functions necesssary for serialization and loading saving to netstate files. */
		virtual void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) ;
		virtual void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) ;
		
	public:
		/*! \brief Standard active switch */
		bool active;

		/*! \brief Standard constructor */
		Device();

		/*! \brief Set device name */
		void set_name(std::string s);

		/*! \brief Get device name */
		std::string get_name();

		/*! \brief Get numeric device id */
		int get_id();

		/*! \brief Flush to file */
		virtual void flush();

		/*! \brief Standard destructor  */
		virtual ~Device();

		/*! Virtual evolve function to be called in central simulation loop in System */
		virtual void evolve() { };

		/*! Virtual execute function to be called at the end of central simulation loop in System */
		virtual void execute() { };
	};

	BOOST_SERIALIZATION_ASSUME_ABSTRACT(Device)

	extern System * sys;
	extern Logger * logger;
}

#endif /*DEVICE_H_*/
