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

#include "ABSConnection.h"

void ABSConnection::init(AurynWeight maxw)
{
	stdp_active = true;
	set_max_weight(maxw);

	tau_post = 100e-3;

	if ( dst->get_post_size() == 0 ) return;

	tr_post = new EulerTrace(dst->get_post_size(),tau_post);
	tr_post->set_target(dst->get_mem_ptr());

	voltage_curve_post 
		= new AurynFloat[ABS_VOLTAGE_CURVE_SIZE];

	set_default_curve();

	set_name("ABSConnection");
}

void ABSConnection::free()
{
	if ( dst->get_post_size() > 0 )
		delete voltage_curve_post;
}

ABSConnection::ABSConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynWeight maxweight , TransmitterType transmitter) 
: DuplexConnection(source, destination, filename, transmitter)
{
	init(maxweight);
}

ABSConnection::ABSConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynWeight maxweight , TransmitterType transmitter, string name) 
: DuplexConnection(source, destination, weight, sparseness, transmitter, name)
{
	init(maxweight);
}

ABSConnection::~ABSConnection()
{
	free();
}

AurynFloat ABSConnection::etamod(NeuronID post)
{
	return 1.0;
}

inline AurynWeight ABSConnection::dw_fwd(NeuronID post)
{
	if ( stdp_active ) {
		NeuronID tr = dst->global2rank(post);
		float v = tr_post->get(tr);
		v -= ABS_VOLTAGE_CURVE_MIN;
		v /= (ABS_VOLTAGE_CURVE_MAX-ABS_VOLTAGE_CURVE_MIN)/ABS_VOLTAGE_CURVE_SIZE;
		int i = (int)v;
		if ( i < 0 || i >= ABS_VOLTAGE_CURVE_SIZE ) return 0.0;
		return etamod(tr)*voltage_curve_post[i];
	}
	else return 0.;
}

inline AurynWeight ABSConnection::dw_bkw(NeuronID pre)
{
	return 0.;
}

void ABSConnection::propagate_forward()
{
	NeuronID * ind = w->get_row_begin(0); // first element of index array
	AurynWeight * data = w->get_data_begin();
	AurynWeight value;
	TransmitterType transmitter = get_transmitter();
	SpikeContainer::const_iterator spikes_end = src->get_spikes()->end();
	// process spikes
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != spikes_end ; ++spike ) {
		for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) {
			value = data[c-ind]; 
			dst->tadd( *c , value , transmitter );
			if (data[c-ind]>0 && data[c-ind]<get_max_weight())
			  data[c-ind] += dw_fwd(*c);
		}
	}
	// if ( sys->get_clock()%10000 == 0 ) 
	// 	cout << scientific << tr_post->get(1) << endl;
	tr_post->follow();
}

void ABSConnection::propagate_backward()
{
}

void ABSConnection::propagate()
{
	// propagate
	propagate_forward();
	propagate_backward();
}

void ABSConnection::load_curve_from_file( const char * filename , double scale ) 
{
	if ( dst->get_post_size() == 0 ) return;

 	stringstream oss;
 	oss << "ABSConnection:: Loading ABS voltage curve from " << filename;
 	logger->msg(oss.str(),NOTIFICATION);

 	ifstream infile (filename);
 	if (!infile) {
 		stringstream oes;
 		oes << "Can't open input file " << filename;
 		logger->msg(oes.str(),ERROR);
 		return;
 	}

	for ( int i = 0 ; i < ABS_VOLTAGE_CURVE_SIZE ; ++i) {
			voltage_curve_post[i] = 0.0; 
	}

 	float value;
 	float voltage;
 	unsigned int count = 0;
 
 	char buffer[256];
 	infile.getline (buffer,256); 
 
 	while ( infile.getline (buffer,256)  )
 	{
 		sscanf (buffer,"%f %f",&voltage,&value);
 		if ( ABS_VOLTAGE_CURVE_MIN <= voltage && voltage <= ABS_VOLTAGE_CURVE_MAX ) {
			int i = (voltage-ABS_VOLTAGE_CURVE_MIN)*ABS_VOLTAGE_CURVE_SIZE/(ABS_VOLTAGE_CURVE_MAX-ABS_VOLTAGE_CURVE_MIN);
			voltage_curve_post[i] = value*scale; 
			// cout << i << " " << value << endl;
 		}
 		count++;
 	}
 
 	infile.close();
}

void ABSConnection::set_default_curve(double fp_low, double fp_middle, double fp_high,double scale)
{
	if ( dst->get_post_size() == 0 ) return;

	for ( int i = 0 ; i < ABS_VOLTAGE_CURVE_SIZE ; ++i) {
		double x = ABS_VOLTAGE_CURVE_MIN+1.0*i/ABS_VOLTAGE_CURVE_SIZE*(ABS_VOLTAGE_CURVE_MAX-ABS_VOLTAGE_CURVE_MIN);
		voltage_curve_post[i] = scale*(x-fp_low)*(x-fp_middle)*(x-fp_high); 
	}
}

