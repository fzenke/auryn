/* 
* Copyright 2014-2015 Friedemann Zenke
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
#include <fstream>
#include <string>

using namespace std;

class System;

/*! \brief Abstract base class for all Monitor objects.
 * 
 * This Class constitutes the base class for all Monitors in Auryn. Per default it openes a single text file for writing (outfile) named by the name supplied in the constructor.
 * Classes inheriting from Monitor have to implement the method propagate. Unlike Checker objects propagate returns void. Use Checker if you need the Monitor to be able to interrupt a run. 
 */

class Monitor
{
private:
	/*! Functions necesssary for serialization and loading saving to netstate files. */
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		virtual_serialize(ar, version);
	}


protected:
	/*! Output filestream to be used in the derived classes */
	ofstream outfile;
	/*! Stores output filename */
	string fname;
	/*! Standard initializer to be called by the constructor */
	void init(string filename);
	/*! Opens a text outputfile -- for binary files redefine this function in derived class. */
	virtual void open_output_file(string filename);
	/*! Standard free function to be called by the destructor - closes the file stream. */
	void free();

	/*! Functions necesssary for serialization and loading saving to netstate files. */
	virtual void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) ;
	virtual void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) ;
	
public:
	/*! Toggles Monitor (in)active */
	bool active;
	/*! Standard constructor */
	Monitor();
	/*! Standard constructor with file name*/
	Monitor(string filename);
	/*! Standard destructor  */
	virtual ~Monitor();
	/*! Virtual propagate function to be called in central simulation loop in System */
	virtual void propagate() = 0;
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Checker)

extern System * sys;
extern Logger * logger;

#endif /*MONITOR_H_*/
