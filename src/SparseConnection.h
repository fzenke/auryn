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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#ifndef SPARSECONNECTION_H_
#define SPARSECONNECTION_H_

#include "auryn_definitions.h"
#include "Connection.h"
#include "System.h"
#include "ComplexMatrix.h"

#include <sstream>
#include <fstream>
#include <stdio.h>
#include <algorithm>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/lognormal_distribution.hpp>

#define WARN_FILL_LEVEL 0.8

using namespace std;

typedef ComplexMatrix<AurynWeight> ForwardMatrix;


/*! \brief The base class to create sparse random connections
 * 
 *  This direct derivative of the virtual Connection type is the most commonly used
 *  class to create connections in Auryn. It makes use of the SimpleMatrix container
 *  to memory efficiently store synaptic weights and to make them easily accessible 
 *  in a feed-forward manner for spike propagation.
 *
 *  Most plastic connection types such as TripletConnection inherit from SparseConnecion
 *  via the intermediate Duplexconnection.
 */

class SparseConnection : public Connection
{
private:
	SpikeContainer * spikes;
	static bool has_been_seeded;
	bool has_been_allocated;
	void init();
	bool init_from_file(const char * filename);

protected:
	void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
	{
		Connection::virtual_serialize(ar,version);
		ar & *w;
	}

	void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
	{
		Connection::virtual_serialize(ar,version);
		ar & *w;
	}

	static boost::mt19937 sparse_connection_gen;

	AurynWeight wmin;
	AurynWeight wmax;
	bool skip_diagonal;

	void free();
	void allocate(AurynLong bufsize);

	
public:
	/*! Switch that toggles for the load_patterns function whether or 
	 * not to use the intensity (gamma) value. Default is false. */
	bool patterns_ignore_gamma; 
	/*! The every_pre parameter allows to skip presynaptically over pattern IDs 
	 * when loading patterns. Default is 1. This can be useful to 
	 * when loading patterns into the exc->inh connections and 
	 * there significantly less inhibitory cells than exc ones. */
	NeuronID patterns_every_pre;
	/*! The every_post parameter allows to skip postsynaptically over pattern IDs 
	 * when loading patterns. Default is 1. This can be useful to 
	 * when loading patterns into the exc->inh connections and 
	 * there significantly less inhibitory cells than exc ones. */
	NeuronID patterns_every_post;
	/*! Switch that toggles the behavior when loading a pattern to
	 * wrap neuron IDs back onto existing cells via the modulo 
	 * function. */
	bool wrap_patterns;

	/*! A pointer that points per defalt to the SimpleMatrix
	 * that stores the connectinos. */
	ForwardMatrix * w; 

	SparseConnection();
	SparseConnection(const char * filename);
	SparseConnection(NeuronID rows, NeuronID cols);
	SparseConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter = GLUT);
	SparseConnection(SpikingGroup * source, NeuronGroup * destination, const char * filename, TransmitterType transmitter=GLUT);
	SparseConnection(SpikingGroup * source, NeuronGroup * destination, AurynWeight weight, AurynFloat sparseness=0.05, TransmitterType transmitter=GLUT, string name="SparseConnection");
	/*! This constructor tries to clone a connection by guessing all parameters except source and destination from another connection instance. */
	SparseConnection(SpikingGroup * source, NeuronGroup * destination, SparseConnection * con, string name="SparseConnection");
	SparseConnection(SpikingGroup * source, NeuronGroup * destination, AurynWeight weight, AurynFloat sparseness, NeuronID lo_row, NeuronID hi_row, NeuronID lo_col, NeuronID hi_col, TransmitterType transmitter=GLUT);
	virtual ~SparseConnection();

	/*! Is used whenever memory has to be allocated manually. Automatically adjust for number of ranks and for security margin */
	void allocate_manually(AurynLong expected_size);

	/*! This function estimates the required size of the nonzero entry buffer. */
	AurynLong estimate_required_nonzero_entires( AurynLong nonzero , double sigma = 5.);

	/*! This function seeds the generator for all random fill operatios */
	void seed(NeuronID randomseed);

	virtual AurynWeight get(NeuronID i, NeuronID j);
	virtual AurynWeight * get_ptr(NeuronID i, NeuronID j);
	virtual AurynWeight get_data(NeuronID i);
	virtual void set_data(NeuronID i, AurynWeight value);
	/*! Sets a single connection to value if it exists  */
	virtual void set(NeuronID i, NeuronID j, AurynWeight value);
	/*! Sets a list of connection to value if they exists  */
	virtual void set(vector<neuron_pair> element_list, AurynWeight value);
	/*! Synonym for random_data_lognormal  */
	void random_data(AurynWeight mean, AurynWeight sigma); 
	/*! Set weights of all existing connections randomly using a normal distrubtion */
	void random_data_normal(AurynWeight mean, AurynWeight sigma); 
	/*! Set weights of all existing connections randomly using a lognormal distribution */
	void random_data_lognormal(AurynWeight m, AurynWeight s); 
	/*! Sets weights in cols to the same value drewn from a Gaussian distribution  */
	void random_col_data(AurynWeight mean, AurynWeight sigma); 
	/*! Sets all weights of existing connections in a block spanned by the first 4 parameters to the value given. */
	void set_block(NeuronID lo_row, NeuronID hi_row, NeuronID lo_col, NeuronID hi_col, AurynWeight weight);
	/*! Sets all weights of existing connections to the given value. */
	virtual void set_all(AurynWeight weight);

	/*! Scales all weights in the weight matrix with the given value. */
	virtual void scale_all(AurynFloat value);

	/*! Clip weights */
	virtual void clip(AurynWeight lo, AurynWeight hi);

	/*! Sets weights in a upper triangular matrix */
	void set_upper_triangular(AurynWeight weight);

	virtual void sparse_set_data(AurynDouble sparseness, AurynWeight value);

	void connect_random(AurynWeight weight=1.0, float sparseness=0.05, bool skip_diag=false);

	/*! Underlying sparse fill method. Set dist_optimized to false and seed
	 * all ranks the same to get the same matrix independent of the number
	 * of ranks. 
	 */ 
	void connect_block_random(AurynWeight weight, 
			float sparseness, 
			NeuronID lo_row, 
			NeuronID hi_row, 
			NeuronID lo_col, 
			NeuronID hi_col, 
			bool skip_diag=false );
	virtual void finalize();
	bool push_back(NeuronID i, NeuronID j, AurynWeight weight);
	AurynLong get_nonzero();
	void put_pattern(type_pattern * pattern, AurynWeight strength, bool overwrite );
	void put_pattern(type_pattern * pattern1, type_pattern * pattern2, AurynWeight strength, bool overwrite );
	/*! Reads patterns from a .pat file and adds them as Hebbian assemblies onto an existing weight matrix */
	void load_patterns( string filename, AurynWeight strength, bool overwrite = false, bool chainmode = false);
	void load_patterns( string filename, AurynWeight strength, int n, bool overwrite = false, bool chainmode = false);
	virtual void propagate();

	/*! Quick an dirty function that checks if all units on the local rank are connected */
	void sanity_check();

	virtual AurynDouble sum();
	virtual void stats(AurynFloat &mean, AurynFloat &std);

	AurynLong dryrun_from_file(string filename);
	bool write_to_file(ForwardMatrix * m, string filename );
	bool load_from_file(ForwardMatrix * m, string filename, AurynLong data_size = 0 );

	virtual bool write_to_file(string filename);

	/*! \brief Loads weight matrix from a single file
	 *
	 * Since a single file might contain a lot more elements than memory required this
	 * function performs a dry run during which it counts the required number of 
	 * elements. This function should be optimized to avoid on large clusters the complete
	 * hammering of the fileserver. An idea would be to let one rank do all the work
	 * and distribute the established file-counts to all the stations
	 */
	virtual bool load_from_complete_file(string filename);
	virtual bool load_from_file(string filename);

	/*! Sets minimum weight (for plastic connections). */
	virtual void set_min_weight(AurynWeight minimum_weight);
	AurynWeight get_min_weight();

	/*! Sets maximum weight (for plastic connections). */
	virtual void set_max_weight(AurynWeight maximum_weight);
	AurynWeight get_max_weight();

	/*! Returns a vector of ConnectionsID of a block specified by the arguments */
	vector<neuron_pair> get_block(NeuronID lo_row, NeuronID hi_row, NeuronID lo_col, NeuronID hi_col);
	/*! Returns a vector of ConnectionsID of postsynaptic parterns of neuron i */
	vector<neuron_pair> get_post_partners(NeuronID i);
	/*! Returns a vector of ConnectionsID of presynaptic parterns of neuron i */
	vector<neuron_pair> get_pre_partners(NeuronID j);
};

#endif /*SPARSECONNECTION_H_*/
