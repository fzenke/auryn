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

#include "FileModulatedPoissonGroup.h"

void FileModulatedPoissonGroup::init ( string filename )
{
	if ( !evolve_locally() ) return;

	inputfile.open(filename.c_str(),ifstream::in);
	if (!inputfile) {
	  cerr << "Can't open input file " << filename << endl;
	  exit(1);
	}

	ftime = 0;
}

FileModulatedPoissonGroup::FileModulatedPoissonGroup(NeuronID n, 
		string filename ) : PoissonGroup( n , 0.0 ) 
{
	init(filename);
}

FileModulatedPoissonGroup::~FileModulatedPoissonGroup()
{
	inputfile.close();
}

void FileModulatedPoissonGroup::evolve()
{

	AurynDouble t ;
	AurynDouble r ;

	// if there are datapoints in the rate file update linear interpolation
	while (ftime < sys->get_clock() && inputfile.getline(buffer, 256) ) {
		istringstream line ( buffer );

		// save first interpolation point
		ltime = ftime;
		rate_n = get_rate();

		line >> t;
		ftime = t/dt;
		line >> r;

		if ( inputfile.eof() ) // to avoid infinit growth
			rate_m = 0.0;
		else
			rate_m = (r-rate_n)/(ftime-ltime);

	}

	AurynDouble rate = rate_m*(sys->get_clock()-ltime)+rate_n;
	if ( rate > 0.0 ) {
		set_rate(rate);
		PoissonGroup::evolve();
	}
}
