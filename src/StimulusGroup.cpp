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

#include "StimulusGroup.h"

boost::mt19937 StimulusGroup::poisson_gen = boost::mt19937(); 
boost::mt19937 StimulusGroup::order_gen = boost::mt19937(2351301); 
boost::uniform_01<boost::mt19937> StimulusGroup::order_die = boost::uniform_01<boost::mt19937> (order_gen);

void StimulusGroup::init(string filename, StimulusGroupModeType stimulusmode, string outputfile, AurynFloat baserate)
{
	sys->register_spiking_group(this);
	ttl = new AurynTime [get_rank_size()];
	activity = new AurynFloat [get_rank_size()];
	set_baserate(baserate);
	poisson_gen.seed(162346*communicator->rank());
	

	mean_off_period = 1.0 ;
	mean_on_period = 0.2 ;
	stimulus_order = stimulusmode ;

	stimulus_active = false ;
	set_all( 0.0 ); 

	scale = 2.0;
	randomintervals = true;

	background_rate = 0.0;
	bgx  = 0 ;

	binary_patterns = false;

	if ( !outputfile.empty() ) 
	{
		tiserfile.open(outputfile.c_str(),ios::out);
		if (!tiserfile) {
		  stringstream oss;
		  oss << "StimulusGroup:: Can't open output file " << filename;
		  logger->msg(oss.str(),ERROR);
		  exit(1);
		}
		tiserfile.setf(ios::fixed);
		// tiserfile.precision(5); 
	}

	stringstream oss;
	oss << "StimulusGroup:: In mode " << stimulus_order;
	logger->msg(oss.str(),NOTIFICATION);

	cur_stim_index = 0;
	next_action_time = 0;
	active = true;
	off_pattern = -1;

	load_patterns(filename);
}

StimulusGroup::StimulusGroup(NeuronID n, string filename, string outputfile, StimulusGroupModeType stimulusmode, AurynFloat baserate) : SpikingGroup( n, STIMULUSGROUP_LOAD_MULTIPLIER ) // Load multiplier is an empircal value
{
	init(filename, stimulusmode, outputfile, baserate);
}

StimulusGroup::~StimulusGroup()
{
	delete [] ttl;
	tiserfile.close();
}

void StimulusGroup::redraw()
{
	boost::exponential_distribution<> dist(BASERATE);
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(poisson_gen, dist);
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i )
	{
		ttl[i] = sys->get_clock() + (AurynTime)((AurynFloat)die()/((activity[i]+1e-9)*dt));
	}
}

void StimulusGroup::redraw_softstart()
{
	boost::exponential_distribution<> dist(BASERATE);
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(poisson_gen, dist);

	boost::uniform_real<> uniformdist(0, SOFTSTARTTIME );
	boost::variate_generator<boost::mt19937&, boost::uniform_real<> > random(poisson_gen, uniformdist);

	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i )
	{
		ttl[i] = sys->get_clock() + (AurynTime)((AurynFloat)die()/((activity[i]+base_rate)*dt)+random()/dt);
	}
}

void StimulusGroup::set_baserate(AurynFloat baserate)
{
	base_rate = baserate;
	redraw();
	logger->parameter("StimulusGroup:: baserate",baserate);
}

void StimulusGroup::set_maxrate(AurynFloat baserate)
{
	set_baserate(baserate);
}

void StimulusGroup::set_mean_off_period(AurynFloat period)
{
	mean_off_period = period;
	logger->parameter("StimulusGroup:: mean_off_period",mean_off_period);
}

void StimulusGroup::set_mean_on_period(AurynFloat period)
{
	mean_on_period = period;
	logger->parameter("StimulusGroup:: mean_on_period",mean_on_period);
}

void StimulusGroup::write_sequence_file(AurynDouble time) {
	if ( tiserfile ) {
		tiserfile << time; 
		for ( unsigned int i = 0 ; i < stimuli.size() ; ++i ) {
			tiserfile << "  ";
			if ( ( stimulus_active  && i == cur_stim_index ) || ( !stimulus_active && i == off_pattern ) ) tiserfile << 1; else tiserfile << 0;
		}
		tiserfile  << endl;
	}
}

void StimulusGroup::evolve()
{
	if ( !active ) return;

	if ( stimulus_active ) {
		// detect and push spikes
		boost::exponential_distribution<> dist(BASERATE);
		boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(poisson_gen, dist);
		for ( NeuronID i = 0 ; i < get_rank_size() ; ++i )
		{
			if ( ttl[i] < sys->get_clock() && activity[i]>0.0 )
			{
				push_spike ( i );
				ttl[i] = sys->get_clock() + (AurynTime)((AurynFloat)die()/((activity[i]+base_rate)*dt));
			}
		}
	} else {
		if ( background_rate ) {
			boost::exponential_distribution<> dist(background_rate);
			boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(poisson_gen, dist);

			while ( bgx < get_rank_size() ) {
				push_spike ( bgx );
				AurynDouble r = die();
				bgx += 1+(NeuronID)(r/dt); 
			}
			bgx -= get_rank_size();

		}
	}

	// update stimulus properties
	if ( sys->get_clock() >= next_action_time ) {
		write_sequence_file(dt*(sys->get_clock()));

		if ( stimulus_active ) {
			set_all( 0.0 ); // turn off currently active stimulus 
			stimulus_active = false ;

			if ( randomintervals ) {
				boost::exponential_distribution<> dist(1./mean_off_period);
				boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > die(order_gen, dist);
				next_action_time = sys->get_clock() + (AurynTime)(max(0.0,die())/dt);
			} else {
				next_action_time = sys->get_clock() + (AurynTime)(mean_off_period/dt);
			}
		} else {
			if ( active ) {

				// choose stimulus
				switch ( stimulus_order ) {
					case MANUAL:
					break;
					case SEQUENTIAL:
						cur_stim_index = (cur_stim_index+1)%stimuli.size();
					break;
					case SEQUENTIAL_REV:
						--cur_stim_index;
						if ( cur_stim_index <= 0 ) 
							cur_stim_index = stimuli.size() - 1 ;
					break;
					case RANDOM:
					default:
						double draw = order_die();
						double cummulative = 0; // TODO make this less greedy and do not compute this every draw
						cur_stim_index = 0;
						// cout.precision(5);
						// cout << " draw " << draw <<  endl;
						for ( unsigned int i = 0 ; i < probabilities.size() ; ++i ) {
							cummulative += probabilities[i];
							// cout << cummulative << endl;
							if ( draw <= cummulative ) {
								cur_stim_index = i;
								break;
							}
						}
					break;
				}
				set_active_pattern( cur_stim_index );
				stimulus_active = true;

				if ( randomintervals ) {
					boost::normal_distribution<> dist(mean_on_period,mean_on_period/3);
					boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(order_gen, dist);
					next_action_time = sys->get_clock() + (AurynTime)(max(0.0,die())/dt);
				} else {
					next_action_time = sys->get_clock() + (AurynTime)(mean_on_period/dt);
				}
			}
		}
		write_sequence_file(dt*(sys->get_clock()+1));
	}
}

void StimulusGroup::set_activity(NeuronID i, AurynFloat val)
{
	activity[i] = max((double)val,1e-9);
}

void StimulusGroup::set_all(AurynFloat val)
{
	for ( unsigned int i = 0 ; i < get_rank_size() ; ++i )
		activity[i] = val;
}

AurynFloat StimulusGroup::get_activity(NeuronID i)
{
	if ( localrank(i) )
		return activity[global2rank(i)];
	else 
		return 0;
}

void StimulusGroup::load_patterns( string filename )
{
		ifstream fin (filename.c_str());
		if (!fin) {
			stringstream oss;
			oss << "StimulusGroup:: "
			<< "There was a problem opening file "
			<< filename
			<< " for reading."
			<< endl;
			logger->msg(oss.str(),ERROR);
			throw AurynOpenFileException();
		}

		char buffer[256];
		string line;

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
					stringstream oss;
					oss << "StimulusGroup:: Read pattern " 
						<< stimuli.size() 
						<< " with pattern size "
						<< total_pattern_size
						<< " ( "
						<< pattern.size()
						<< " on rank )";
					logger->msg(oss.str(),DEBUG);

					stimuli.push_back(pattern);
					pattern.clear();
					total_pattern_size = 0;
				}
				continue;
			}

			stringstream iss (line);
			NeuronID i ;
			iss >> i ;
			if ( localrank( i ) ) {
				pattern_member pm;
				pm.gamma = 1 ;
				iss >>  pm.gamma ;
				pm.i = global2rank( i ) ;
				pattern.push_back( pm ) ;
			}
			total_pattern_size++;
		}

		fin.close();

		// initializing all probabilities as a flat distribution
		flat_distribution();

		stringstream oss;
		oss << "StimulusGroup:: Finished loading " << stimuli.size() << " patterns";
		logger->msg(oss.str(),NOTIFICATION);
}

void StimulusGroup::set_pattern_activity(unsigned int i)
{
	type_pattern current = stimuli[i];
	type_pattern::iterator iter;

	if ( binary_patterns ) { 
		for ( iter = current.begin() ; iter != current.end() ; ++iter )
		{
			set_activity(iter->i,scale);
		}
	} else { 
		for ( iter = current.begin() ; iter != current.end() ; ++iter )
		{
			set_activity(iter->i,scale*iter->gamma);
		}
	}
}

void StimulusGroup::set_pattern_activity(unsigned int i,AurynFloat setrate)
{
	type_pattern current = stimuli[i];
	type_pattern::iterator iter;

	for ( iter = current.begin() ; iter != current.end() ; ++iter )
	{
		set_activity(iter->i,setrate);
	}
}


void StimulusGroup::set_active_pattern(unsigned int i)
{
	stringstream oss;
	oss << "StimulusGroup:: Setting active pattern " << i ;
	logger->msg(oss.str(),DEBUG);

	set_all( 0.0 );
	if ( i < stimuli.size() ) {
		set_pattern_activity(i);
	}
	redraw();
}

void StimulusGroup::set_distribution( vector<double> probs )
{
	stringstream oss;
	oss << "StimulusGroup: Set distribution [";
	for ( unsigned int i = 0 ; i < stimuli.size() ; ++i ) {
		probabilities[i] = probs[i];
		oss << " " << probabilities[i];
	}
	oss << " ]";
	logger->msg(oss.str(),NOTIFICATION);

	normalize_distribution();
}

vector<double> StimulusGroup::get_distribution( )
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
	stringstream oss;
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
	logger->msg(oss.str(),DEBUG);
}

vector<type_pattern> * StimulusGroup::get_patterns()
{
	return &stimuli;
}

void StimulusGroup::set_next_action_time( double time ) {
	next_action_time = sys->get_clock() + time/dt;
}
