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

#ifndef SPARSECONNECTION_H_
#define SPARSECONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Connection.h"
#include "System.h"
#include "ComplexMatrix.h"

#include <sstream>
#include <fstream>
#include <stdio.h>
#include <algorithm>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/lognormal_distribution.hpp>

#define WARN_FILL_LEVEL 0.8

namespace auryn {

typedef ComplexMatrix<AurynWeight> ForwardMatrix;


/*! \brief The base class to create sparse random connections
 * 
 *  This direct derivative of the virtual Connection type is the most commonly used
 *  class to create connections in Auryn. It makes use of the ComplexMatrix container
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
	/*! \brief Used to test loading of a weight matrix and to count number of elements in wmat files */
	AurynLong dryrun_from_file(string filename);

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

	/*! /brief Static random number generator used for random connect */
	static boost::mt19937 sparse_connection_gen;

	/*! /brief Minimum allowed weight value
	 *
	 * This property is stored for Connection objects with plasticity. The value can be set and accessed 
	 * with the setters get_min_weight() and set_min_weight(x) */
	AurynWeight wmin;

	/*! /brief Maximum allowed weight value
	 *
	 * This property is stored for Connection objects with plasticity. The value can be set and accessed 
	 * with the setters get_max_weight() and set_max_weight(x) */
	AurynWeight wmax;

	/*! Switch that specifies whether or not to skip diagonal elemens during random connect. This 
	 * is usefull for random connects to exclude autapses from the connections. */
	bool skip_diagonal;

	void free();

	/*! Allocates memory for a given sparse connectivity matrix. Usually ComplexMatrix or ComplexMatrix */
	void allocate(AurynLong bufsize);


	
public:
	/*! \brief Switch that toggles for the load_patterns function whether or 
	 * not to use the intensity (gamma) value. Default is false. */
	bool patterns_ignore_gamma; 

	/*! \brief The every_pre parameter allows to skip presynaptically over pattern IDs 
	 * when loading patterns. Default is 1. This can be useful to 
	 * when loading patterns into the exc->inh connections and 
	 * there significantly less inhibitory cells than exc ones. */
	NeuronID patterns_every_pre;

	/*! \brief The every_post parameter allows to skip postsynaptically over pattern IDs 
	 * when loading patterns. Default is 1. This can be useful to 
	 * when loading patterns into the exc->inh connections and 
	 * there significantly less inhibitory cells than exc ones. */
	NeuronID patterns_every_post;

	/*! \brief Switch that toggles the behavior when loading a pattern to
	 * wrap neuron IDs back onto existing cells via the modulo 
	 * function. */
	bool wrap_patterns;

	/*! \brief A pointer that points per default to the ComplexMatrix
	 * that stores the connectinos. */
	ForwardMatrix * w; 

	/*! \brief Empty constructor which should not be used -- TODO should be deprecated at some point. */
	SparseConnection();
	/*! \brief Load from wmat file constructor which should not be used -- TODO should be deprecated at some point. */
	SparseConnection(const char * filename);

	/*! \brief Constructor for manual filling. */
	SparseConnection(NeuronID rows, NeuronID cols);

	SparseConnection(SpikingGroup * source, NeuronGroup * destination, const char * filename, TransmitterType transmitter=GLUT);

	/*! \brief Default constructor which sets up a random sparse matrix with fixed weight between the source and destination group. 
	 *
	 * The constructor takes the weight and sparseness as secondary arguments. The latter allows Auryn to 
	 * allocate the approximately right amount of memory inadvance. It is good habit to specify at time of initialization also 
	 * a connection name and the transmitter type. Both can be set separately with set_transmitter and set_name if the function call gets
	 * too long and ugly. 
	 * A connection name is often handy during debugging and the transmitter type is a crucial for obvious resons ...  
	 * */
	SparseConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			AurynWeight weight, 
			AurynDouble sparseness=0.05, 
			TransmitterType transmitter=GLUT, 
			string name="SparseConnection"
			);

	/*! \brief This constructor tries to clone a connection by guessing all parameters 
	 * except source and destination from another connection instance. */
	SparseConnection(SpikingGroup * source, NeuronGroup * destination, SparseConnection * con, string name="SparseConnection");

	/*! \brief Sparse block constructor
	 *
	 *  This constructor initializes the connection with random sparse weights, but only fills a "block" as specified instead of the
	 *  entire matrix. */
	SparseConnection(SpikingGroup * source, NeuronGroup * destination, AurynWeight weight, AurynDouble sparseness, NeuronID lo_row, NeuronID hi_row, NeuronID lo_col, NeuronID hi_col, TransmitterType transmitter=GLUT);
	
	/*! \brief The default destructor */
	virtual ~SparseConnection();

	/*! \brief Is used whenever memory has to be allocated manually. Automatically adjusts for number of ranks and for security margin. */
	void allocate_manually(AurynLong expected_size);

	/*! \brief This function estimates the required size of the nonzero entry buffer. 
	 *
	 * It's typicall used internally or when you know what you are doing. */
	AurynLong estimate_required_nonzero_entires( AurynLong nonzero , double sigma = 5.);

	/*! \brief This function seeds the pseudo random number generator for all random fill operatios. */
	void seed(NeuronID randomseed);

	/*! \brief Returns weight value of a given element if it exists */
	virtual AurynWeight get(NeuronID i, NeuronID j);

	/*! \brief Returns pointer to given weight element if it exists. Returns NULL if element does not exist. */
	virtual AurynWeight * get_ptr(NeuronID i, NeuronID j);

	/*! \brief Returns weight value of a given element referenced by index in the data array. */
	virtual AurynWeight get_data(NeuronID i);

	/*! \brief Sets weight value of a given element referenced by its index in the data array. */
	virtual void set_data(NeuronID i, AurynWeight value);

	/*! \brief Sets a single connection to value if it exists  */
	virtual void set(NeuronID i, NeuronID j, AurynWeight value);

	/*! \brief Sets a list of connection to value if they exists  */
	virtual void set(std::vector<neuron_pair> element_list, AurynWeight value);

	/*! \brief Synonym for random_data  */
	void random_data(AurynWeight mean, AurynWeight sigma); 

	/*! \brief Set weights of all existing connections randomly using a normal distrubtion */
	void random_data_normal(AurynWeight mean, AurynWeight sigma); 

	/*! \brief Set weights of all existing connections randomly using a lognormal distribution */
	void random_data_lognormal(AurynWeight m, AurynWeight s); 

	/*! \brief Initialize with random binary at wlo and whi.  
	 * \param wlo The lower weight value. 
	 * \param whi The higher weight value.
	 * \param prob the probability for the higher value. */
	void init_random_binary(AurynFloat prob=0.5, AurynWeight wlo=0.0, AurynWeight whi=1.0); 

	/*! \brief Sets weights in cols to the same value drewn from a Gaussian distribution  */
	void random_col_data(AurynWeight mean, AurynWeight sigma); 

	/*! \brief Sets all weights of existing connections in a block spanned by the first 4 parameters to the value given. */
	void set_block(NeuronID lo_row, NeuronID hi_row, NeuronID lo_col, NeuronID hi_col, AurynWeight weight);

	/*! \brief Sets all weights of existing connections to the given value. */
	virtual void set_all(AurynWeight weight);

	/*! \brief Scales all weights in the weight matrix with the given value. */
	virtual void scale_all(AurynFloat value);

	/*! \brief Clip weights */
	virtual void clip(AurynWeight lo, AurynWeight hi);

	/*! \brief Sets weights in a upper triangular matrix */
	void set_upper_triangular(AurynWeight weight);

	/*! \brief Sets a sparse random subset of connection elements wight the given value */
	virtual void sparse_set_data(AurynDouble sparseness, AurynWeight value);

	/*! \brief Connect src and dst SpikingGroup and NeuronGroup randomly with given sparseness 
	 *
	 * This function should be usually called from the constructor directly. */
	void connect_random(AurynWeight weight=1.0, AurynDouble sparseness=0.05, bool skip_diag=false);

	/*! \brief Underlying sparse fill method. 
	 *
	 * Set dist_optimized to false and seed all ranks the same to get the same
	 * matrix independent of the number of ranks. 
	 * Called internally or when you know what you are doing.
	 */ 
	void connect_block_random(AurynWeight weight, 
			AurynDouble sparseness, 
			NeuronID lo_row, 
			NeuronID hi_row, 
			NeuronID lo_col, 
			NeuronID hi_col, 
			bool skip_diag=false );

	/*! \brief Finalizes connection after random or manual initialization of the weights.
	 *
	 * Essentially pads zeros or non-existing elements at the end of ComplexMatrix. 
	 * Called interally or after manually filling matrices. */
	virtual void finalize();
	
	/*! \brief Pushes a single element to the ComplexMatrix.
	 *
	 * Note that Auryn sparse matrices need to be filled row by row in column increasing order (similar 
	 * to writing in a text document). Hence, usually this function is called internally during weight initialization 
	 * through a connect_random method. However, it can also be invoked manually to build custum
	 * weight matrices on the fly. The recommended method however is to build specific weight matrices
	 * in another high level programming language such as Python and to save the in the Auryn specific 
	 * Matrix Market format which can then be loaded using load_from_file or load_from_complete_file 
	 * methods. */
	bool push_back(NeuronID i, NeuronID j, AurynWeight weight);

	/*! \brief Returns number of nonzero elements in this SparseConnection */
	AurynLong get_nonzero();

	/*! \brief Puts cell assembly to existing sparse weights 
	 *
	 * TODO add more explanation here.*/
	void put_pattern(type_pattern * pattern, AurynWeight strength, bool overwrite );

	/*! \brief Puts cell assembly or synfire pattern to existing sparse weights
	 *
	 * TODO add more explanation here.*/
	void put_pattern(type_pattern * pattern1, type_pattern * pattern2, AurynWeight strength, bool overwrite );

	/*! \brief Reads patterns from a .pat file and adds them as Hebbian assemblies onto an existing weight matrix */
	void load_patterns( string filename, AurynWeight strength, bool overwrite = false, bool chainmode = false);

	/*! \brief Reads first n patterns from a .pat file and adds them as Hebbian assemblies onto an existing weight matrix */
	void load_patterns( string filename, AurynWeight strength, int n, bool overwrite = false, bool chainmode = false);

	/*! \brief Internally used propagate method
	 *
	 * This method propagates spikes in the main simulation loop. Should usually not be called directly by the user.*/
	virtual void propagate();

	/*! \brief Quick an dirty function that checks if all units on the local rank are connected */
	void sanity_check();

	/*! \brief Computes sum of all weight elements in the Connection */
	virtual AurynDouble sum();

	/*! \brief Computes mean and variance of weights in default weight matrix
	 *
	 * Returns mean and variance of default weight matrix (typically referenced as w 
	 * in a given SparseConnection */
	virtual void stats(AurynDouble &mean, AurynDouble &std);

	/*! \brief Computes mean and variance of weights for matrix state zid
	 */
	virtual void stats(AurynDouble &mean, AurynDouble &std, NeuronID zid);


	/*! \brief Writes rank specific weight matrix on the same rank to a file
	 *
	 * This function writes all synaptic weights from the specified weight matrix which are stored on the same rank 
	 * to a Matrix Market file in real coordinate format. The file can later be read with load_from_file to continue 
	 * a simulation or can be processed offline using standard tools such as MATLAB or Python.
	 */
	bool write_to_file(ForwardMatrix * m, string filename );

	/*! \brief Writes rank specific default weight matrix on the same rank to a file
	 *
	 * This call is a shortcut for write_to_file(w, filename) where w is the default weight matrix of
	 * the underlying SparseConnectoin.
	 */
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

	/*! \brief Loads weight matrix from Matrix Market (wmat) file
	 *
	 * This function expects an Auryn readable Matrix Market file such as generated by
	 * write_to_file methods which only includes weight elements which belong on this very rank.
	 * To load wmat files containing all weights indepent of rank use the load_from_complete_file 
	 * method. Note that these methods only store information of the first element of a ComplexMatrix. 
	 * To store all informtion of a ComplexMatrix use the mechanisms in place to save the network state
	 * which are implemented in the System class. */
	virtual bool load_from_file(string filename);

	/*! \brief Loads weight matrix from Matrix Market (wmat) file to specified weight matrix
	 *
	 * This function expects an Auryn readable Matrix Market file such as generated by
	 * write_to_file methods which only includes weight elements which belong on this very rank.
	 * To load wmat files containing all weights indepent of rank use the load_from_complete_file 
	 * method. */
	bool load_from_file(ForwardMatrix * m, string filename, AurynLong data_size = 0 );

	/*! \brief Sets minimum weight (for plastic connections). */
	virtual void set_min_weight(AurynWeight minimum_weight);

	/*! \brief Gets minimum weight (for plastic connections). */
	AurynWeight get_min_weight() {
		return wmin;
	};

	/*! \brief Sets maximum weight (for plastic connections). */
	virtual void set_max_weight(AurynWeight maximum_weight);

	/*! \brief Gets maximum weight (for plastic connections). */
	AurynWeight get_max_weight() {
		return wmax;
	};

	/*! \brief Returns a vector of ConnectionsID of a block specified by the arguments */
	std::vector<neuron_pair> get_block(NeuronID lo_row, NeuronID hi_row, NeuronID lo_col, NeuronID hi_col);

	/*! \brief Returns a vector of ConnectionsID of postsynaptic parterns of neuron i */
	std::vector<neuron_pair> get_post_partners(NeuronID i);

	/*! \brief Returns a vector of ConnectionsID of presynaptic parterns of neuron i */
	std::vector<neuron_pair> get_pre_partners(NeuronID j);
};

} // namespace 

#endif /*SPARSECONNECTION_H_*/
