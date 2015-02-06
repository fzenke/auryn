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

#include "System.h"
#include "Logger.h"
#include "EulerTrace.h"
#include "LinearTrace.h"


int main(int ac, char* av[]) 
{
	// BEGIN Global stuff
	mpi::environment env(ac, av);
	mpi::communicator world;
	communicator = &world;

	char strbuf [255];
	sprintf(strbuf, "%s/%s.log", ".", "test_traces" );
	string logfile = strbuf;
	logger = new Logger(logfile,world.rank(),PROGRESS,EVERYTHING);

	sys = new System(&world);
	// END Global stuff

	double tau = 20e-3;
	EulerTrace * tr_euler = new EulerTrace(2,tau);
	LinearTrace * tr_linear = new LinearTrace(2,tau,sys->get_clock_ptr());

	for ( int i = 0 ; i < 10000 ; ++i ) {
		if ( i % 332 == 0 ) { 
			tr_euler->inc(0);
			tr_linear->inc(0);
		}
		if ( i % 100 == 0 ) {
			cout << i*1e-4 << " " << tr_euler->get(0) 
						   << " " << tr_linear->get(0) 
						   << " " << tr_euler->get(0) 
						   << " " << tr_linear->get(0) 
				 << endl;
		}

		tr_euler->evolve();
		tr_linear->evolve();

		sys->step();
	}

	for ( int i = 10000 ; i < 200000 ; ++i ) {
		if ( i % 1024 == 0 ) { 
			tr_euler->inc(0);
			tr_linear->inc(0);
		}
		if ( i % 1050 == 0 ) {
			cout << i*1e-4 << " " << tr_euler->get(0) 
						   << " " << tr_linear->get(0) 
						   << " " << tr_euler->get(0) 
						   << " " << tr_linear->get(0) 
				 << endl;
		}
		tr_euler->evolve();
		tr_linear->evolve();
		sys->step();
	}

	return 0;
}
