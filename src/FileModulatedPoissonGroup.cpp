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

#include "FileModulatedPoissonGroup.h"

using namespace auryn;

void FileModulatedPoissonGroup::init ( std::string filename )
{
	if ( !evolve_locally() ) return;

	inputfile.open(filename.c_str(),std::ifstream::in);
	if (!inputfile) {
	  std::cerr << "Can't open input file " << filename << std::endl;
	  std::exit(1);
	}

	ftime = 0;
}

FileModulatedPoissonGroup::FileModulatedPoissonGroup(NeuronID n, 
		std::string filename ) : PoissonGroup( n , 0.0 ) 
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
	while (ftime < auryn::sys->get_clock() && inputfile.getline(buffer, 256) ) {
		std::stringstream line ( buffer );

		// save first interpolation point
		ltime = ftime;
		rate_n = get_rate();

		line >> t;
		ftime = (AurynTime) (t/dt+0.5);
		line >> r;

		if ( ftime < auryn::sys->get_clock() || inputfile.eof() ) { // if the recently read point is already in the past -> reinit interpolation
			rate_m = 0.0;
			rate_n = r;
			set_rate(r);
		} else { // compute linear interpolation
			rate_m = (r-rate_n)/(ftime-ltime);
		}
	}

	AurynDouble rate = rate_m*(auryn::sys->get_clock()-ltime)+rate_n;

	if ( last_rate != rate ) { // only redraw when rate changes
		set_rate(rate);
	}

	if ( rate ) 
		PoissonGroup::evolve();

	last_rate = rate;
}
