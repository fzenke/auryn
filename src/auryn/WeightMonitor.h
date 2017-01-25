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

#ifndef WEIGHTMONITOR_H_
#define WEIGHTMONITOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Monitor.h"
#include "System.h"
#include "SparseConnection.h"
#include <fstream>
#include <iomanip>

namespace auryn {


/*! RecordingMode determines the default recording behavior of the monitor.
 * The modes SINGLE, DATARANGE and ELEMENTLIST (default) record weight values
 * from single synapses while GROUPS records the statistics over groups of 
 * synapses */
enum RecordingMode { 
	SINGLE, ///< The entire Monitor will record from a single synapse specified at initialization.
	DATARANGE, ///< The Monitor will record from a range of synapses specified at initialization.
	ELEMENTLIST, ///< The Monitor records from selected synapses stored in a list. This is the default behavior.
	GROUPS /*!< This mode is added in versions >0.4.1 and allows to record summary statistics of 
			 synapses between neural groups/patterns. */
};

/*! Determines how pattern files are interpreted for loading. 
 * ALLTOALL will add connections between all possible pattern combinations, 
 * whereas ASSEMBLIES_ONLY restricts recording to inside each pattern. */
enum PatternMode { ALLTOALL, ASSEMBLIES_ONLY};

/*! \brief Monitors the evolution of a single or a set of weights 
 *
 * This class perform online monitoring of synaptic weights in a Connection. The class supports different recording modes
 * for single individual weights or multiple weights. Output is written to a human readable text file.
 * 
 * \todo Implement a new data_index based element list which can deal with Complex Matrix states in a clean fashion
 */ 

class WeightMonitor : protected Monitor
{
protected:
	SparseConnection * src;
	ForwardMatrix * mat;
	RecordingMode recordingmode;
	NeuronID elem_i;
	NeuronID elem_j;
	AurynTime ssize;
	std::vector<AurynLong> * element_list;
	std::vector<NeuronID> group_indices;
	void init(SparseConnection * source, NeuronID i, NeuronID j, string filename, AurynTime interval);

	void record_single_synapses();
	void record_synapse_groups();

	std::vector<type_pattern> * load_patfile( string filename, unsigned int maxpat );
	
public:
	/*! \brief Default constructor
	 *
	 * This initializes the WeightMonitor and specifies and output file and a sampling interval.
	 * Per default WeightMonitor is in multi recording mode. That means it maintains a list of pointers to weights which you would like
	 * to monitor. This list is initally empty and needs to be populated after initializating with the appropriate add ... functions.
	 * \param source The connection object to record from
	 * \param filename The file to record to
	 * \param interval The sampling interval in simulation interval in seconds
	 * */
	WeightMonitor(SparseConnection * source, string filename, AurynDouble interval=10.0);

	/*! \brief Specific weight matrix constructor
	 *
	 * This initializes the WeightMonitor and specifies and output file and a sampling interval.
	 * The constructor does not specify which weights to monitor.
	 * interpreted as range i..j in the data array of the source Connection. Note that in DATARANGE mode the Monitor could in principle
	 * also monitor zero connections. However in SINGLE it records only non-zero elements.
	 * \param source The connection object to record from
	 * \param filename The file to record to
	 * \param m A pointer to the weight matrix to record from
	 * \param interval The sampling interval in simulation interval in seconds
	 * */
	WeightMonitor(SparseConnection * source, ForwardMatrix * m, string filename, AurynDouble interval=10.0);

	/*! \brief Single weight constructor 
	 *
	 * This initializes the WeightMonitor and specifies and output file and a sampling interval. 
	 * It moreover puts the WeightMonitor in RecordingMode mode SINGLE in which it records a single synaptic connection
	 * specified by i and j.
	 * If mode is set to DATARANGE instead, the arguments i and j are instead interpreted as 
	 * range i..j in the data index array of the source Connection. 
	 * \param source The connection object to record from
	 * \param i Parameter i (either row position of synapse in WeightMatrix or start index in data array depending on mode)
	 * \param j Parameter j (either col position of synapse in WeightMatrix or stop index in data array depending on mode) 
	 * \param filename The file to record to
	 * \param interval The sampling interval in simulation interval (1s  default) 
	 * \param mode The recording mode (one of RecordingMode datatype)
	 * \param z The z state the complex matrix state to record from
	 * */
	WeightMonitor(SparseConnection * source, NeuronID i, NeuronID j, string filename, AurynDouble interval=10.0, RecordingMode mode = SINGLE, StateID z = 0);

	virtual ~WeightMonitor();

	/*! \brief propagate function used by System class internally */
	void execute();

	/*! \brief Sets target matrix to read from. */
	void set_mat(ForwardMatrix * m);

	/*! \brief Adds a single element to the recording list which is identified by its data index and z-state. 
	 * \param index The index in the data array of this sparse matrix. 
	 * \param z The id of the complex matrix state to record from. */
	void add_to_list_by_data_index( AurynLong index, StateID z = 0 );

	/*! \brief Adds a single element to the recording list which is identified by a pointer. */
	void add_to_list( AurynWeight * ptr );

	/*! \brief Adds a single element identified matrix coordinates (row,col,z) to the recording list. */
	void add_to_list( NeuronID i, NeuronID j, StateID z = 0 );

	/*! \brief Adds a list vector<neuron_pair> vec the the recording list. 
	 *
	 * Such a list can for instance be generated by a SparseConnection with the
	 * get_block function. 
	 */
	void add_to_list( std::vector<neuron_pair> vec , string label = "");
	
	/*! \brief Adds number of elements to the recording list that are equally spaced in the 
	 * data vector of the connection matrix. 
	 *
	 * This effectively corresponds to a number of random 
	 * connection matrix elements (if a random matrix is used). 
	 * The second argument can be used to 
	 * specify which complex synapse state should be used for this operation 
	 * when complex synapses are used.
	 */
	void add_equally_spaced( NeuronID number, NeuronID z=0 );

	/*! \brief Adds connections inside a pattern and between patterns. Since such lists
	 * can become large quickly the amount of patterns and connections for each
	 * pattern to be monitored can be limited by maxcon and maxpat.
	 * \param filename The filename of the .pat file
	 * \param maxcon maximum number of connection per pattern or between patterns
	 * \param maxpat maximum number of patterns to read from .pat file
	 * \param patmod ALLTOALL means from all patterns to all patterns. ASSEMBLIES_ONLY
	 *		only adds connections inside single patterns.
	 */
	void load_pattern_connections(string filename, unsigned int maxcon = 5, unsigned int maxpat = 10, PatternMode patmod = ALLTOALL);
	void load_pattern_connections(string filename_pre, string filename_post, unsigned int maxcon = 5, unsigned int maxpat = 10, PatternMode patmod = ALLTOALL);

	/*! \brief Adds connections from a given range in the sparse matrix data array */
	void load_data_range(AurynLong i, AurynLong j);
};

} 

#endif /*WEIGHTMONITOR_H_*/
