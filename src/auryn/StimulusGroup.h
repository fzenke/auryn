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

#ifndef STIMULUSGROUP_H_

#define STIMULUSGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "SpikingGroup.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#define SOFTSTARTTIME 0.1

namespace auryn {


/*! \brief Provides a poisson stimulus at random intervals in one or more
 *         predefined subsets of the group that are read from a file. */
class StimulusGroup : public SpikingGroup
{
private:
	AurynTime * clk;

	/*! Internal name for the stimfile (tiser stands for time series). */
	std::fstream tiserfile;

	AurynFloat base_rate;

	int off_pattern;


protected:
	AurynTime * ttl;

	AurynFloat * activity;

	/*! Stimulus order */
	StimulusGroupModeType stimulus_order ;

	/*! Foreground Poisson field pointer */
	NeuronID fgx;

	/*! Background Poisson field pointer */
	NeuronID bgx;

	/*! stimulus probabilities */
	std::vector<double> probabilities ;

	/*! pseudo random number generators */
	static boost::mt19937 poisson_gen; 

	/*! generates info for what stimulus is active. Is supposed to give the same result on all nodes (hence same seed required) */
	static boost::mt19937 order_gen; 
	static boost::uniform_01<boost::mt19937> order_die; 



	/*! \brief next stimulus time requiring change in rates */
	AurynTime next_action_time ;

	/*! \brief last stimulus time requiring change in rates */
	AurynTime last_action_time ;

	/*! Standard initialization */
	void init(StimulusGroupModeType stimulusmode, string stimfile, AurynFloat baserate);

	/*! Draw all Time-To-Live (ttls) typically after changing the any of the activiteis */
	virtual void redraw();

	/*! write current stimulus to stimfile */
	void write_stimulus_file(AurynDouble time);

	/*! Read current stimulus status from stimfile */
	void read_next_stimulus_from_file(AurynDouble &time, int &active, int &stimulusid );

	/*! Sets the activity for a given unit on the local rank. Activity determines the freq as baserate*activity */
	void set_activity( NeuronID i, AurynFloat val=0.0 );

	/*! allow silence/background activity periods */
	AurynFloat mean_off_period ;

	/*! mean presentation time  */
	AurynFloat mean_on_period ;

	AurynFloat curscale;
	
public:
	/*! \brief Current stimulus index 
	 *
	 * Do not write this variable. */
	int cur_stim_index ;

	/*! \brief Current stimulus active 
	 *
	 * Only read this state. */
	bool stimulus_active;


	/*! \brief Vector containing all the stimuli. */
	std::vector<type_pattern> stimuli;

	/*! \brief This is by how much the pattern gamma value is multiplied. The resulting value gives the x-times baseline activation */
	AurynFloat scale;

	/*! \brief Switches to more efficient algorithm which ignores the gamma value */
	bool binary_patterns;

	/*! \brief Enables a finite refractory time specified in AurynTime (only works for non-binary-pattern mode. */
	AurynTime refractory_period;

	/*! \brief Determines if the Group is using random activation intervals */
	bool randomintervals;

	/*! \brief Determines if the Group is using random activation intensities */
	bool randomintensities;

	/*! \brief Play random Poisson noise with this rate on all channels 
	 * when no stim is active. */
	AurynDouble background_rate;

	/*! \brief Switch for background firing during stimulus. */
	bool background_during_stimulus;

	/*! \brief Default constructor */
	StimulusGroup(NeuronID n, string filename, string stimfile, StimulusGroupModeType stimulusmode=RANDOM, AurynFloat baserate=0.0 );

	/*! \brief Constructor without stimfile. Patterns can be loaded afterwards using the load_patterns method. */
	StimulusGroup(NeuronID n, string stimfile, StimulusGroupModeType stimulusmode=RANDOM, AurynFloat baserate=0.0 );

	virtual ~StimulusGroup();
	/*! \brief Standard virtual evolve function */
	virtual void evolve();
	/*! \brief Sets the baserate that is the rate at 1 activity */
	void set_baserate(AurynFloat baserate);
	void set_maxrate(AurynFloat baserate); // TODO remove deprecated

	/*! \brief Sets the stimulation mode. Can be any of StimulusGroupModeType (MANUAL,RANDOM,SEQUENTIAL,SEQUENTIAL_REV). */
	void set_stimulation_mode(StimulusGroupModeType mode);

	/*! \brief Sets sets the activity of all units */
	void set_all( AurynFloat val=0.0 );

	/*! \brief Seeds the random number generator for all stimulus groups of the simulation. */
	void seed( int rndseed );

	/*! \brief Gets the activity of unit i */
	AurynFloat get_activity(NeuronID i);

	/*! \brief Loads stimulus patterns from a designated file given */
	void load_patterns( string filename );

	/*! \brief Set mean quiet interval between consecutive stimuli */
	void set_mean_off_period(AurynFloat period);

	/*! \brief Set mean on period */
	void set_mean_on_period(AurynFloat period);

	/*! \brief Function that loops over the stimulus/pattern vector and sets the activity verctor to the gamma values given with the pattern. */
	void set_pattern_activity( unsigned int i );

	/*! \brief Function that loops over the stimulus/pattern vector and sets the activity verctor to the given value. */
	void set_pattern_activity( unsigned int i, AurynFloat setval );

	/*! \brief This function is called internally and sets the activity level to a given active stimulus
	 *
	 * @param i the index of the pattern to set the activity to
	 */
	void set_active_pattern( unsigned int i );

	/*! \brief This function is called internally and sets the activity level to a given active stimulus
	 *
	 * @param i The index of the pattern to set the activity to
	 * @param default_value The value to assign to the activity values which are not specified in the pattern file. 
	 * Typically this corresponds to some background value.
	 */
	void set_active_pattern( unsigned int i, AurynFloat default_value);

	void set_next_action_time(double time);

	/*! \brief Setter for pattern probability distribution */
	void set_distribution ( std::vector<double> probs );
	/*! \brief Getter for pattern probability distribution */
	std::vector<double> get_distribution ( );
	/*! \brief Getter for pattern i of the probability distribution */
	double get_distribution ( int i );

	/*! \brief returns the last action (stim on/off) time in units of AurynTime */
	AurynTime get_last_action_time();

	/*! \brief returns the next action (stim on/off) time in units of AurynTime */
	AurynTime get_next_action_time();

	/*! \brief returns the index of the current (or last -- if not active anymore) active stimulus */
	unsigned int get_cur_stim();

	/*! \brief Returns true if currently a stimulus is active and false otherwise. */
	bool get_stim_active();

	/*! Initialized distribution to be flat */
	void flat_distribution( );
	/*! Normalizes the distribution */
	void normalize_distribution( );

	std::vector<type_pattern> * get_patterns();

};

}

#endif /*STIMULUSGROUP_H_*/
