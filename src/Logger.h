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

#ifndef LOGGER_H_
#define LOGGER_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

#define CERRLEVEL WARNING

using namespace std;

/*! Enum type for significance level of a given message send to the logger */
enum LogMessageType { EVERYTHING, DEBUG, NOTIFICATION, SETTINGS, PROGRESS, WARNING, ERROR, NONE };

/*! \brief A generic logger class that logs to screen and a log-file.
 *
 * Logs message to console and a log-file. What goes where can be adjusted by
 * the LogMessageType.
 */
class Logger
{
private:
	string fname;
	ofstream outfile;
	int local_rank;
	LogMessageType console_out;
	LogMessageType file_out;

	string last_message;

	
public:
	Logger(string filename, int rank, LogMessageType console = PROGRESS, LogMessageType file = NOTIFICATION );
	void msg( string text, LogMessageType type=NOTIFICATION, bool global=false, int line=-1, string srcfile="" );
	void parameter( string name, double value );
	void parameter( string name, int value );
	void parameter( string name, string value );
	void set_rank(int rank);
	virtual ~Logger();
};

#endif /*LOGGER_H_*/
