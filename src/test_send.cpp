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

#include <boost/mpi.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <boost/serialization/string.hpp>
#include "SpikeContainer.h"

namespace mpi = boost::mpi;
using namespace std;

int msgtag(int x, int y, int s) {
	return x*s+y;
}

int main(int argc, char* argv[]) 
{
	int len = 10;

	mpi::environment env(argc, argv);
	mpi::communicator world;

	mpi::request sendreqs[10]; // TODO max num of requests
	mpi::request recvreqs[10]; // TODO max num of requests
	SpikeContainer out_msg;
	SpikeContainer msg[10];
	for (int i = 0 ; i < len ; ++i) 
		out_msg.push_back(i+len*world.rank());

	// cout << "This is out msg " << world.rank() << endl;
	// out_msg.print_spikes();
	// cout << " Sending ... " << endl;
	
	for (int i = 0 ; i < world.size()-1 ; ++i) {
		int dst = (world.rank()+i+1)%world.size();
		int src = dst;
		int tag = msgtag(world.rank(),dst,world.size());
		cout << "isend: " << world.rank() << "::"  << dst << " tag " << tag << endl;
		sendreqs[i] = world.isend(dst, tag, out_msg);
		tag = msgtag(src,world.rank(),world.size());
		cout << "irecv: " << world.rank() << "::"  << src << " tag " << tag << endl;
		recvreqs[i] = world.irecv(src, tag, msg[i]);
	}

	mpi::wait_all(recvreqs, recvreqs + world.size()-1);

	if (world.rank()==1) {
		for (int i = 0 ; i < world.size()-1 ; ++i) {
			cout << "This is rank " << world.rank() << endl;
			msg[i].print_spikes();
		}
	}
  return 0;
}

