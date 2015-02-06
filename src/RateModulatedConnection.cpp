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

#include "RateModulatedConnection.h"

void RateModulatedConnection::init() 
{
	rate_target = 5;
	rate_estimate = rate_target;
	rate_estimate_tau = 1.0;
	rate_estimate_decay_mul = exp(-dt/rate_estimate_tau);

	rate_modulation_exponent = 2;
	rate_modulation_mul = 1.0;
	rate_modulating_group = NULL;


	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank
	set_modulating_group(dst);


	// Synaptic traces
	eta = 1e-3; // learning rate

}


RateModulatedConnection::RateModulatedConnection(const char * filename) 
: SparseConnection(filename)
{
	init();
}

RateModulatedConnection::RateModulatedConnection(SpikingGroup * source, NeuronGroup * destination, 
		TransmitterType transmitter) 
: SparseConnection(source, destination, transmitter)
{
}

RateModulatedConnection::RateModulatedConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename , 
		TransmitterType transmitter) 
: SparseConnection(source, destination, filename, transmitter)
{
	init();
}


RateModulatedConnection::RateModulatedConnection(NeuronID rows, NeuronID cols) 
: SparseConnection(rows,cols)
{
	init();
}

RateModulatedConnection::RateModulatedConnection( SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		TransmitterType transmitter, string name) 
: SparseConnection(source,destination,weight,sparseness,transmitter, name)
{
	init();
}

void RateModulatedConnection::free()
{
}

void RateModulatedConnection::propagate_forward()
{
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) { // c = post index
			NeuronID * ind = w->get_ind_begin(); // first element of index array
			AurynWeight * data = w->get_data_begin();
			AurynWeight value = data[c-ind]; 
			transmit( *c , rate_modulation_mul*value );
		}
	}
}


void RateModulatedConnection::propagate()
{
	propagate_forward();
}

void RateModulatedConnection::evolve()
{
	// compute the averages
	if ( rate_modulating_group==NULL ) return;

	rate_estimate *= rate_estimate_decay_mul;
	rate_estimate += 1.0*rate_modulating_group->get_spikes()->size()/rate_estimate_tau/rate_modulating_group->get_size();
	rate_modulation_mul += eta*tanh(rate_target-rate_estimate);
	if ( rate_modulation_mul < 0 ) rate_modulation_mul = 0;
	if ( rate_modulation_mul > 10 ) rate_modulation_mul = 10;
}

RateModulatedConnection::~RateModulatedConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}


void RateModulatedConnection::set_modulating_group(SpikingGroup * group)
{
	if ( group->evolve_locally() )
		rate_modulating_group = group;
}

void RateModulatedConnection::set_eta(AurynFloat value)
{
	eta = value;
}

void RateModulatedConnection::stats(AurynFloat &mean, AurynFloat &std)
{
	SparseConnection::stats(mean,std);
	mean *= rate_modulation_mul;
	std *= rate_modulation_mul;
}

bool RateModulatedConnection::write_to_file(string filename)
{

	stringstream oss;
	oss << filename << ".cstate";

	ofstream outfile;
	outfile.open(oss.str().c_str(),ios::out);
	if (!outfile) {
	  cerr << "Can't open output file " << filename << endl;
	  throw AurynOpenFileException();
	}

	boost::archive::text_oarchive oa(outfile);
	oa << rate_estimate ;
	oa << rate_modulation_mul ;

	outfile.close();

	return SparseConnection::write_to_file(filename);
}

bool RateModulatedConnection::load_from_file(string filename)
{

	stringstream oss;
	oss << filename << ".cstate";
	ifstream infile (oss.str().c_str());

	if (!infile) {
		stringstream oes;
		oes << "Can't open input file " << filename;
		logger->msg(oes.str(),ERROR);
		throw AurynOpenFileException();
	}

	boost::archive::text_iarchive ia(infile);
	ia >> rate_estimate;
	ia >> rate_modulation_mul;

	infile.close();

	return SparseConnection::load_from_file(filename);
}
