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
*/

#ifndef MODULATEDPAIRINTERACTIONCONNECTION_H_
#define MODULATEDPAIRINTERACTIONCONNECTION_H_

#define WINDOW_MAX_SIZE 60000

#include "auryn_definitions.h"
#include "DuplexConnection.h"
#include "EulerTrace.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_int.hpp>

using namespace std;


class modulatedPairInteractionConnection : public DuplexConnection
{

protected:
	AurynWeight w_max;
	AurynTime filetime;
	AurynTime * last_spike_pre;
	AurynTime * last_spike_post;
	AurynFloat mods;
	AurynFloat newmods;
	ifstream modulation_file;    

	inline AurynWeight dw_fwd(NeuronID post);
	inline AurynWeight dw_bkw(NeuronID pre);

	inline void propagate_forward();
	inline void propagate_backward();

	static boost::mt19937 gen; 
	boost::uniform_int<int> * dist;
	boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > * die;


public:
	AurynFloat * window_pre_post;
	AurynFloat * window_post_pre;
    AurynFloat gmod;

	bool stdp_active;

	modulatedPairInteractionConnection(SpikingGroup * source, NeuronGroup * destination, const char * modulation_filename,
			const char * filename, 
			AurynWeight maxweight=1. , TransmitterType transmitter=GLUT);

	modulatedPairInteractionConnection(SpikingGroup * source, NeuronGroup * destination,  const char * modulation_filename,
			AurynWeight weight, AurynFloat sparseness=0.05,
			AurynWeight maxweight=1. , TransmitterType transmitter=GLUT, string name="modulatedPairInteractionConnection");
	virtual ~modulatedPairInteractionConnection();
	void init(AurynWeight maxw);
	void free();
    void check_modulation_file(const char * modulation_file);

	void load_window_from_file( const char * filename , double scale = 1. );
	void set_exponential_window ( double Aplus = 1e-3, double tau_plus = 20e-3, double Aminus = -1e-3, double tau_minus = 20e-3);
	void set_box_window ( double Aplus = 1e-3, double tau_plus = 20e-3, double Aminus = -1e-3, double tau_minus = 20e-3);
	void set_floor_terms( double pre_post = 0.0, double post_pre = 0.0 );

	virtual void propagate();

};

#endif /*PAIRINTERACTIONCONNECTION_H_*/
