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

#include "StimulusGroup.h"

using namespace auryn;

boost::mt19937 StimulusGroup::poisson_gen = boost::mt19937(); 
boost::mt19937 StimulusGroup::order_gen = boost::mt19937(); 
boost::uniform_01<boost::mt19937> StimulusGroup::order_die = boost::uniform_01<boost::mt19937> (order_gen);

void StimulusGroup::init(StimulusGroupModeType stimulusmode, std::string stimfile, AurynFloat baserate)
{
	auryn::sys->register_spiking_group(this);
	ttl = new AurynTime [get_rank_size()];

	refractory_period = 1; // initialize with a default of one timestep (avoids two spikes in same time bin)

	activity = new AurynFloat [get_rank_size()];
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) activity[i] = 0.0;
	set_baserate(baserate);

	seed(2351301);

	set_stimulation_mode(stimulusmode);

	stimulus_active = false ;
	set_all( 0.0 ); 

	randomintervals = true;
	mean_off_period = 1.0 ;
	mean_on_period = 0.2 ;

	randomintensities = false;
	scale = 1.0; // TODO does this need to be initialized at 2 ?
	curscale = scale;

	background_during_stimulus = false;

	background_rate = 0.0;
	bgx  = 0 ;

	fgx  = 0 ;


	binary_patterns = false;

	// if a filename was supplied and we are 
	// not supposed to be reading from it.
	if ( !stimfile.empty() && stimulus_order != STIMFILE ) 
	{
		tiserfile.open(stimfile.c_str(),std::ios::out);
		tiserfile.setf(std::ios::fixed);
	} else {
		if (stimulus_order==STIMFILE) {
			tiserfile.open(stimfile.c_str(),std::ios::in);
		}
	}

	if (!tiserfile) {
		std::stringstream oss;
		oss << "StimulusGroup:: Cannot open stimulus file " 
			<< stimfile
			<< " for ";
		if (stimulus_order==STIMFILE) oss << "reading.";
		else oss << "writing.";

		auryn::logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	std::stringstream oss;
	oss << "StimulusGroup:: " 
		<< "size " << get_size() << " "  
		<< "(mode " << stimulus_order << ") ";
	auryn::logger->msg(oss.str(),INFO);

	cur_stim_index = 0;
	next_action_time = 0;
	last_action_time = 0;
	active = true;
	off_pattern = -1;

}

StimulusGroup::StimulusGroup(NeuronID n, std::string filename, std::string stimfile, StimulusGroupModeType stimulusmode, AurynFloat baserate) : SpikingGroup( n ) 
{
	init(stimulusmode, stimfile, baserate);
	load_patterns(filename);
}

StimulusGroup::StimulusGroup(NeuronID n, std::string stimfile, StimulusGroupModeType stimulusmode, AurynFloat baserate) : SpikingGroup( n ) // Load multiplier is an empirical value
{
	init(stimulusmode, stimfile, baserate);
}

StimulusGroup::~StimulusGroup()
{
	delete [] ttl;
	tiserfile.close();
}

void StimulusGroup::redraw()
{
	boost::exponential_distribution<> dist(1.0);
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(poisson_gen, dist);
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i )
	{
		if (activity[i]>0) 
			ttl[i] = auryn::sys->get_clock() + (AurynTime)((AurynFloat)die()/(activity[i]*auryn_timestep));
	}
}

void StimulusGroup::set_baserate(AurynFloat baserate)
{
	base_rate = baserate;
	redraw();
	auryn::logger->parameter("StimulusGroup:: baserate",baserate);
}

void StimulusGroup::set_maxrate(AurynFloat baserate)
{
	set_baserate(baserate);
}

void StimulusGroup::set_mean_off_period(AurynFloat period)
{
	mean_off_period = period;
	auryn::logger->parameter("StimulusGroup:: mean_off_period",mean_off_period);
}

void StimulusGroup::set_mean_on_period(AurynFloat period)
{
	mean_on_period = period;
	auryn::logger->parameter("StimulusGroup:: mean_on_period",mean_on_period);
}

void StimulusGroup::write_stimulus_file(AurynDouble time) {
	if ( tiserfile && stimulus_order != STIMFILE ) {
		tiserfile 
			<< time
			<< " ";
		if ( stimulus_active || off_pattern > -1 ) tiserfile << "1 "; else tiserfile << "0 ";
		tiserfile 
			<< cur_stim_index
			<< std::endl;
	}
}

void StimulusGroup::read_next_stimulus_from_file(AurynDouble &time, int &active, int &stimulusid ) {
	char buffer[256];
	if ( tiserfile.getline (buffer,256) ) {
		sscanf (buffer,"%lf %i %i",&time,&active,&stimulusid);
	} else {
		time = auryn::sys->get_time()+1000; // TODO this is a bit weird as a condition but should do the job
		active = 0;
		stimulusid = 0;
	}
}

void StimulusGroup::evolve()
{
	if ( !active ) return;


	if ( stimulus_active ) { // during active stimulation

		if ( binary_patterns ) { // binary patterns
			// detect and push spikes

			boost::exponential_distribution<> dist(curscale);
			boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(poisson_gen, dist);

			type_pattern current = stimuli[cur_stim_index];

			if ( !background_during_stimulus )
				bgx = get_rank_size();

			while ( bgx < get_rank_size() || fgx < current.size() ) {
				if ( fgx < current.size() && current.at(fgx).i < bgx ) {
					push_spike ( current.at(fgx).i );
					AurynDouble r = die();
					fgx += 1+(NeuronID)(r/auryn_timestep);
				} else {
					push_spike ( bgx );

					boost::exponential_distribution<> bg_dist(background_rate);
					boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > bg_die(poisson_gen, bg_dist);
					AurynDouble r = bg_die();
					bgx += 1+(NeuronID)(r/auryn_timestep); 
				}
			}
			if ( background_during_stimulus )
				bgx -= get_rank_size();

			if ( fgx >= current.size() )
				fgx -= current.size();

		} else { // non-binary patterns --- using time-to-live TTL mechanism
			boost::exponential_distribution<> dist(1.0);
			boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(poisson_gen, dist);

			for ( NeuronID i = 0 ; i < get_rank_size() ; ++i )
			{
				if ( ttl[i] < auryn::sys->get_clock() && activity[i]>0.0 )
				{
					push_spike ( i );
					ttl[i] = auryn::sys->get_clock() + refractory_period + (AurynTime)((AurynFloat)die()*(1.0/(activity[i]*auryn_timestep)-refractory_period));
				}
			}
		}
	} else { // while stimulation is off
		if ( background_rate ) {
			boost::exponential_distribution<> dist(background_rate);
			boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(poisson_gen, dist);

			while ( bgx < get_rank_size() ) {
				push_spike ( bgx );
				AurynDouble r = die();
				bgx += 1+(NeuronID)(r/auryn_timestep); 
			}
			bgx -= get_rank_size();
		}
	}

	// update stimulus properties
	if ( auryn::sys->get_clock() >= next_action_time ) { // action required
		last_action_time = next_action_time; // store last time before updating next_action_time

		if ( stimuli.size() == 0 ) {
			set_next_action_time(10); // TODO make this a bit smarter at some point -- i.e. could send this to the end of time 
			return;
		}

		write_stimulus_file(auryn_timestep*(auryn::sys->get_clock()));

		// if we have variable rate stimuli update curscale otherwise set to scale 
		// this is only needed for binary stimuli -- otherwise the change is done in
		// set_pattern_activity
		if ( randomintensities && binary_patterns ) {
			curscale = scale*(AurynFloat)order_die();
		} else {
			curscale = scale;
		}

		if ( stimulus_order == STIMFILE )  {
			AurynDouble t = 0.0;
			int a,i;
			while ( t <= auryn::sys->get_time() ) {
				read_next_stimulus_from_file(t,a,i);
				next_action_time = (AurynTime) (t/auryn_timestep);
				if (a==0) stimulus_active = true; 
					else stimulus_active = false;
				cur_stim_index = i;
				// std::cout << auryn::sys->get_time() << " " << t << " " << a << " " << i << std::endl;
			}
		} else { // we have to generate stimulus times

			if ( stimulus_active ) { // stimulus was active and going inactive now

				if ( !binary_patterns )
					set_all( 0.0 ); // turns off currently active stimulus 
				stimulus_active = false ;

				if ( randomintervals ) {
					boost::exponential_distribution<> dist(1./mean_off_period);
					boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(order_gen, dist);
					next_action_time = auryn::sys->get_clock() + (AurynTime)(std::max(0.0,die())/auryn_timestep);
				} else {
					next_action_time = auryn::sys->get_clock() + (AurynTime)(mean_off_period/auryn_timestep);
				}
			} else { // stimulus was not active and is going active now
				if ( active && stimuli.size() ) { // the group is active and there are stimuli in the array

					// chooses stimulus according to schema specified in stimulusmode
					double draw, cummulative;
					switch ( stimulus_order ) {
						case RANDOM:
							// TODO make this less greedy 
							// and do not compute this every draw
							draw = order_die();
							cummulative = 0; 
							cur_stim_index = 0;
							// std::cout.precision(5);
							// std::cout << " draw " << draw <<  std::endl;
							for ( unsigned int i = 0 ; i < probabilities.size() ; ++i ) {
								cummulative += probabilities[i];
								// std::cout << cummulative << std::endl;
								if ( draw <= cummulative ) {
									cur_stim_index = i;
									break;
								}
							}
						break;
						case SEQUENTIAL:
							cur_stim_index = (cur_stim_index+1)%stimuli.size();
						break;
						case SEQUENTIAL_REV:
							--cur_stim_index;
							if ( cur_stim_index <= 0 ) 
								cur_stim_index = stimuli.size() - 1 ;
						break;
						case MANUAL:
						default:
						break;
					}

					if ( !binary_patterns )
						set_active_pattern( cur_stim_index );
					stimulus_active = true;

					if ( randomintervals && stimulus_order != STIMFILE ) {
						boost::normal_distribution<> dist(mean_on_period,mean_on_period/3);
						boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(order_gen, dist);
						next_action_time = auryn::sys->get_clock() + (AurynTime)(std::max(0.0,die())/auryn_timestep);
					} else {
						next_action_time = auryn::sys->get_clock() + (AurynTime)(mean_on_period/auryn_timestep);
					}
				}
			}
			write_stimulus_file(auryn_timestep*(auryn::sys->get_clock()+1));
		}
	}
}

void StimulusGroup::set_activity(NeuronID i, AurynFloat val)
{
	activity[i] = val;
}

void StimulusGroup::set_all(AurynFloat val)
{
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i )
		set_activity(i,val);
}

AurynFloat StimulusGroup::get_activity(NeuronID i)
{
	if ( localrank(i) )
		return activity[global2rank(i)];
	else 
		return 0;
}

void StimulusGroup::load_patterns( std::string filename )
{
	std::ifstream fin (filename.c_str());
	if (!fin) {
		std::stringstream oss;
		oss << "StimulusGroup:: "
		<< "There was a problem opening file "
		<< filename
		<< " for reading."
		<< std::endl;
		auryn::logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	char buffer[256];
	std::string line;

	stimuli.clear();

	type_pattern pattern;
	int total_pattern_size = 0;
	while(!fin.eof()) {

		line.clear();
		fin.getline (buffer,255);
		line = buffer;

		if (line[0] == '#') continue;
		if (line == "") { 
			if ( total_pattern_size > 0 ) {
				std::stringstream oss;
				oss << "StimulusGroup:: Read pattern " 
					<< stimuli.size() 
					<< " with pattern size "
					<< total_pattern_size
					<< " ( "
					<< pattern.size()
					<< " on rank )";
				auryn::logger->msg(oss.str(),VERBOSE);

				stimuli.push_back(pattern);
				pattern.clear();
				total_pattern_size = 0;
			}
			continue;
		}

		std::stringstream iss (line);
		NeuronID i ;
		iss >> i ;
		if ( localrank( i ) ) {
			pattern_member pm;
			pm.gamma = 1.0 ;
			iss >>  pm.gamma ;
			pm.i = global2rank( i ) ;
			pattern.push_back( pm ) ;
		}
		total_pattern_size++;
	}

	fin.close();

	// initializing all probabilities as a flat distribution
	flat_distribution();

	std::stringstream oss;
	oss << "StimulusGroup:: Finished loading " << stimuli.size() << " patterns";
	auryn::logger->msg(oss.str(),NOTIFICATION);
}

void StimulusGroup::set_pattern_activity(unsigned int i)
{
	type_pattern current = stimuli[i];
	type_pattern::iterator iter;

	AurynFloat addrate = 0.0;
	if ( background_during_stimulus ) 
		addrate = background_rate;

	AurynFloat curscale = scale;
	if ( randomintensities ) {
		boost::exponential_distribution<> dist(1.);
		boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(order_gen, dist);
		curscale *= (AurynFloat)die();
	}

	for ( iter = current.begin() ; iter != current.end() ; ++iter )
	{
		set_activity(iter->i,curscale*iter->gamma+addrate);
	}
}

void StimulusGroup::set_pattern_activity(unsigned int i, AurynFloat setrate)
{
	type_pattern current = stimuli[i];
	type_pattern::iterator iter;

	for ( iter = current.begin() ; iter != current.end() ; ++iter )
	{
		set_activity(iter->i,setrate);
	}
}


void StimulusGroup::set_active_pattern(unsigned int i, AurynFloat default_value)
{
	std::stringstream oss;
	oss << "StimulusGroup:: Setting active pattern " << i ;
	auryn::logger->msg(oss.str(),VERBOSE);

	set_all( default_value );
	if ( i < stimuli.size() ) {
		set_pattern_activity(i);
	}
	redraw();
}

void StimulusGroup::set_active_pattern(unsigned int i)
{
	set_active_pattern(i, background_rate);
}


void StimulusGroup::set_distribution( std::vector<double> probs )
{
	for ( unsigned int i = 0 ; i < stimuli.size() ; ++i ) {
		probabilities[i] = probs[i];
	}

	normalize_distribution();

	std::stringstream oss;
	oss << "StimulusGroup: Set distribution [";
	for ( unsigned int i = 0 ; i < stimuli.size() ; ++i ) {
		oss << " " << probabilities[i];
	}
	oss << " ]";
	auryn::logger->msg(oss.str(),NOTIFICATION);
}

std::vector<double> StimulusGroup::get_distribution( )
{
	return probabilities;
}

double StimulusGroup::get_distribution( int i )
{
	return probabilities[i];
}

void StimulusGroup::flat_distribution( ) 
{
	for ( unsigned int i = 0 ; i < stimuli.size() ; ++i ) {
		probabilities.push_back(1./((double)stimuli.size()));
	}
}

void StimulusGroup::normalize_distribution()
{
	std::stringstream oss;
	oss << "StimulusGroup: Normalizing distribution [";
	double sum = 0 ;
	for ( unsigned int i = 0 ; i < stimuli.size() ; ++i ) {
		sum += probabilities[i];
	}

	// normalize vector 
	for ( unsigned int i = 0 ; i < stimuli.size() ; ++i ) {
		probabilities[i] /= sum;
		oss << " " << probabilities[i];
	}

	oss << " ]";
	auryn::logger->msg(oss.str(),VERBOSE);
}

std::vector<type_pattern> * StimulusGroup::get_patterns()
{
	return &stimuli;
}

void StimulusGroup::set_next_action_time( double time ) {
	next_action_time = auryn::sys->get_clock() + time/auryn_timestep;
}

void StimulusGroup::set_stimulation_mode( StimulusGroupModeType mode ) {
	stimulus_order = mode ;
}

void StimulusGroup::seed(int rndseed)
{
	order_gen.seed(sys->get_synced_seed()); // has to be seeded identically on all ranks!

	unsigned int rnd = rndseed + sys->get_seed(); // adds salt to make it different across ranks
	std::stringstream oss;
	oss << "StimulusGroup:: " 
		<< "seeding Poisson generator with " 
		<< rnd;
	auryn::logger->msg(oss.str(),VERBOSE);
	
	poisson_gen.seed(rnd); // is now drawn differently but reproducibly so for each rank
}

AurynTime StimulusGroup::get_last_action_time()
{
	return last_action_time;
}

AurynTime StimulusGroup::get_next_action_time()
{
	return next_action_time;
}

bool StimulusGroup::get_stim_active()
{
	return stimulus_active;
}

unsigned int StimulusGroup::get_cur_stim()
{
	return cur_stim_index;
}
