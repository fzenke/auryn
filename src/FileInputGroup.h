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

#ifndef FILEINPUTGROUP_H_
#define FILEINPUTGROUP_H_

#include <fstream>
#include <sstream>

#include "auryn_definitions.h"
#include "System.h"
#include "SpikingGroup.h"

/*! \brief Reads files from a ras file and emits them as SpikingGroup in a simulation.
 *
 * When the FileInputGroup reaches the end of a designated ras file it can
 * depending on the settings start over again at the beginning or do nothing.
 * This is controlled by the loop directive.  In addition to that it is
 * possible to specify a certain delay between loops.
 */
class FileInputGroup : public SpikingGroup
{
private:
	AurynTime ftime;
	NeuronID lastspike;
	bool therewasalastspike;
	bool playinloop;
	AurynTime dly;
	AurynTime off;
	ifstream spkfile;
	const char * fname;
	char buffer[255];
	void init(const char * filename );
	
public:
	bool active ;
	FileInputGroup(NeuronID n, const char * filename );
	FileInputGroup(NeuronID n, const char * filename , bool loop, AurynFloat delay );
	virtual ~FileInputGroup();
	virtual void evolve();

};

#endif /*FILEINPUTROUP_H_*/
