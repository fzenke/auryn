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


#include "auryn_definitions.h"
// #include "SpikeContainer.h"


int main(int ac, char* av[]) 
{
	SpikeContainer * sc = new SpikeContainer();

	cout << "Loading 1 2 3 ... " << endl;

	sc->push_back(1);
	sc->push_back(2);
	sc->push_back(3);

	cout << "Reading ";

	for ( NeuronID * iter = sc->begin() ; iter != sc->end() ; ++iter )
		cout << *iter << " ";

	cout << endl;

	sc->clear();

	cout << "Loading 5 6 7 ... " << endl;

	sc->push_back(5);
	sc->push_back(6);
	sc->push_back(7);

	cout << "Reading ";

	for ( NeuronID * iter = sc->begin() ; iter != sc->end() ; ++iter )
		cout << *iter << " ";

	cout << endl;

	delete sc;

	return 0;
}
